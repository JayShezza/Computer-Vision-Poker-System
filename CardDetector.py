from ultralytics import YOLO
import cv2
from collections import Counter, defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
#import CardMonteCarloEquity

model = YOLO("runs/detect/train2/weights/best.pt")
model.to("cuda")

cap = cv2.VideoCapture(0)

cluster_first_seen = {}
cluster_printed = set()
track_label_history = defaultdict(list)
track_last_seen = {}
track_positions = {}
track_confidences = {}

MIN_CONF = 0.35
MAJ_CONF = 0.6
MIN_STABLE_FRAMES = 5
MAX_MISSING_FRAMES = 45
MAJ_WINDOW = 30

DBSCAN_EPS = 500
DBSCAN_MIN_SAMPLES = 1

frame_count = 0


def format_card(label: str):
    return label[:-1] + label[-1].lower()


def majority_list(items):
    if not items:
        return None
    most_common, count = Counter(items).most_common(1)[0]
    if count / len(items) < MAJ_CONF:
        return None
    return most_common

def deduplicate_cards(active_cards):
    seen = {}
    for card in active_cards:
        label = card["label"]
        if label not in seen or card["conf"] > seen[label]["conf"]:
            seen[label] = card
    return list(seen.values())


def current_game_state():
    """Returns stable cards with their bounding box centers."""
    active_cards = []

    for card_id in list(track_last_seen.keys()):
        if frame_count - track_last_seen[card_id] > MAX_MISSING_FRAMES:
            continue
        labels = track_label_history[card_id]
        if len(labels) < MIN_STABLE_FRAMES:
            continue
        label = majority_list(labels)
        if label and card_id in track_positions:
            active_cards.append({
                "id": card_id,
                "label": format_card(label),
                "center": track_positions[card_id],
                "conf": track_confidences.get(card_id, 0.0)
            })

    return active_cards


def cluster_cards(active_cards):
    """
    Uses DBSCAN to spatially cluster cards, then classifies each cluster:
      - 2 cards    → Player Hand
      - 3–5 cards  → Community Cards
      - other      → Unknown Group
    Returns a list of cluster dicts.
    """
    if not active_cards:
        return []

    centers = np.array([[c["center"][0] * 1.0, c["center"][1] * 3.0] for c in active_cards])

    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(centers)
    labels = db.labels_

    clusters = defaultdict(list)
    for card, cluster_id in zip(active_cards, labels):
        clusters[cluster_id].append(card)

    result = []
    for cluster_id, cards in clusters.items():
        count = len(cards)
        card_names = sorted(set(c["label"] for c in cards))

        if count == 2:
            role = "Player Hand"
        elif 3 <= count <= 5:
            role = "Community Cards"
        else:
            role = "Unknown Group"

        result.append({
            "cluster_id": cluster_id,
            "role": role,
            "cards": card_names,
            "centers": [c["center"] for c in cards]
        })

    return result


def draw_cluster_overlay(frame, clusters):
    """Draw a labeled bounding box around each detected cluster."""
    for cluster in clusters:
        track_confidences[track_id] = conf
        centers = cluster["centers"]
        if not centers:
            continue

        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        x1, y1 = max(0, min(xs) - 60), max(0, min(ys) - 90)
        x2, y2 = min(frame.shape[1], max(xs) + 60), min(frame.shape[0], max(ys) + 90)

        color = (0, 200, 255) if cluster["role"] == "Player Hand" else \
                (0, 255, 100) if cluster["role"] == "Community Cards" else \
                (180, 180, 180)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cluster['role']}: {', '.join(cluster['cards'])}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return frame


while True:
    plugged, frame = cap.read()
    if not plugged:
        break

    results = model.track(frame, conf=MIN_CONF, verbose=False, persist=True)

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        ids = boxes.id
        clss = boxes.cls
        confs = boxes.conf
        xyxy = boxes.xyxy  # Get box coordinates for center calculation

        if ids is not None:
            for track_id_tensor, cls_tensor, conf_tensor, box in zip(ids, clss, confs, xyxy):
                track_id = int(track_id_tensor.item())
                cls_id = int(cls_tensor.item())
                conf = float(conf_tensor.item())

                if conf < MIN_CONF:
                    continue

                # Card coords center
                x1, y1, x2, y2 = box.tolist()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                track_positions[track_id] = (cx, cy)

                label = model.names[cls_id]
                track_label_history[track_id].append(label)
                track_last_seen[track_id] = frame_count
                track_label_history[track_id] = track_label_history[track_id][-MAJ_WINDOW:]

    active_cards = current_game_state()
    active_cards = deduplicate_cards(active_cards)
    clusters = cluster_cards(active_cards)

    annotated = results[0].plot() if results else frame.copy()
    annotated = draw_cluster_overlay(annotated, clusters)

    # HUD
    current_keys = set()
    for cluster in clusters:

        centers = [c["center"] for c in active_cards]
        if len(centers) >= 2:
            xs = [c[0] for c in centers]
            ys = [c[1] for c in centers]
            print(f"X spread: {max(xs) - min(xs)}px, Y spread: {max(ys) - min(ys)}px")

        if cluster["role"] not in ("Player Hand", "Community Cards"):
            continue
        key = f"{cluster['role']}:{','.join(sorted(cluster['cards']))}"
        current_keys.add(key)
        if key not in cluster_first_seen:
            cluster_first_seen[key] = frame_count
        elif frame_count - cluster_first_seen[key] == 30:
            print(f"{cluster['role']}: {cluster['cards']}")

    for key in list(cluster_first_seen):
        if key not in current_keys:
            del cluster_first_seen[key]

    cv2.imshow("Card Tracker", annotated)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()