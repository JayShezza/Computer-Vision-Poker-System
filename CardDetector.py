from ultralytics import YOLO
import cv2
from collections import Counter, defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
from CardMonteCarloEquity import monte_carlo_equity

model = YOLO("../best.pt")
model.to("cuda")

cap = cv2.VideoCapture(0)

cluster_first_seen = {}
cluster_printed = set()
track_label_history = defaultdict(list)
track_last_seen = {}
track_positions = {}
track_confidences = {}
equity_pending = {}
equity_printed = set()
last_equity_results = None

# ---- GLOBALS ---- #
MIN_CONF = 0.35               # How confident does YOLO need to be in its prediction
MAJ_CONF = 0.6                # How many times does the same prediction get made in the MAJ_WINDOW
MIN_STABLE_FRAMES = 5         # Cards must be stable for this many frames minimum
MAX_MISSING_FRAMES = 45       # How many frames before cards are removed
MAJ_WINDOW = 30               # How many frames to look over cards
EQUITY_CALC_DELAY_FRAMES = 45 # frames before detected board is sent to equity calculations
MIN_PLAYER_HANDS = 2          # Minimum holes needed before sending to equity
MC_SIMULATIONS = 10000        # How many times to simulate board (higher == longer time && more accurate) 10,000 ~ 3s
DBSCAN_EPS = 500              # Radius in pixels to cluster
DBSCAN_MIN_SAMPLES = 1        # Min number of cards per cluster

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
    """Uses DBSCAN to cluster cards by position and assigns them a role."""
    if not active_cards:
        return []

    centers = np.array([[c["center"][0] * 1.0, c["center"][1] * 5.0] for c in active_cards])

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

        if cluster["role"] == "Player Hand":
            color = (0, 150, 255)
        elif cluster["role"] == "Community Cards":
            color = (0, 255, 150)
        else:
            color = (150, 150, 150)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cluster['role']}: {', '.join(cluster['cards'])}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return frame


def draw_equity_overlay(frame, equity_results, player_hands):
    """Draw Equity results onto cards on screen"""
    if not equity_results or not player_hands:
        return frame

    y_offset = 30
    cv2.putText(frame, "Equity:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 55, 0), 2)
    y_offset += 30

    for result in equity_results:
        cards_str = ", ".join(result["hole_cards"])
        equity = result["equity"]
        wins = result["wins"]
        ties = result["ties"]
        losses = result["losses"]
        text = f"P{result['player']} [{cards_str}]  Equity: {equity:.1f}%"
        cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 55, 0), 2)
        y_offset += 28

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

    # HUD
    current_keys = set()
    player_hands = [c for c in clusters if c["role"] == "Player Hand"]
    community = [c for c in clusters if c["role"] == "Community Cards"]

    annotated = results[0].plot() if results else frame.copy()
    annotated = draw_cluster_overlay(annotated, clusters)
    annotated = draw_equity_overlay(annotated, last_equity_results, player_hands)



    for cluster in clusters:
        if cluster["role"] not in ("Player Hand", "Community Cards"):
            continue
        key = f"{cluster['role']}:{','.join(sorted(cluster['cards']))}"
        current_keys.add(key)

        if key not in cluster_first_seen:
            cluster_first_seen[key] = frame_count
        elif frame_count - cluster_first_seen[key] == 30:
            print(f"{cluster['role']}: {cluster['cards']}")

    if len(player_hands) >= MIN_PLAYER_HANDS:
        all_cards = sorted(
            [c for cl in player_hands + community for c in cl["cards"]]
        )
        equity_key = "|".join(all_cards)

        if equity_key not in equity_pending:
            equity_pending[equity_key] = frame_count
        elif (
                equity_key not in equity_printed
                and frame_count - equity_pending[equity_key] >= EQUITY_CALC_DELAY_FRAMES
        ):
            # Equity Calcs
            community_cards = community[0]["cards"] if community else []
            hands = [h["cards"] for h in player_hands]
            print(f"[Equity Trigger] Community: {community_cards}, Hands: {hands}")

            last_equity_results = monte_carlo_equity(
                players_hole_cards=hands,
                community_cards=community_cards,
                simulations=MC_SIMULATIONS
            )

            for r in last_equity_results:
                print(f"Player {r['player']} {r['hole_cards']}: {r['equity']}% equity  |  W: {r['wins']} T: {r['ties']} L: {r['losses']}")

            equity_printed.add(equity_key)

    # Invalidate pending equity states that are no longer visible
    for key in list(equity_pending):
        cards_in_key = set(key.split("|"))
        visible_cards = {c for cl in clusters for c in cl["cards"]}
        if not cards_in_key.issubset(visible_cards):
            del equity_pending[key]
            equity_printed.discard(key)  # allow recalculation if cards reappear

    for key in list(cluster_first_seen):
        if key not in current_keys:
            del cluster_first_seen[key]

    cv2.imshow("Card Tracker", annotated)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()