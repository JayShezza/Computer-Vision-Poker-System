from ultralytics import YOLO
import cv2
from collections import Counter, defaultdict

model = YOLO("runs/detect/train2/weights/best.pt")
model.to("cuda")

cap = cv2.VideoCapture(0)

track_label_history = defaultdict(list)
track_last_seen = {}

MIN_CONF = 0.5
MAJ_CONF = 0.6
MIN_STABLE_FRAMES = 1
MAX_MISSING_FRAMES = 45


frame_count = 0


def format_card(label: str):
    """Format card names"""
    return label[:-1] + label[-1].lower()


def majority_list(items):
    """Majority voting function"""
    if not items:
        return None
    most_common, count = Counter(items).most_common(1)[0]
    if count / len(items) < MAJ_CONF:
        return None
    return most_common

    return Counter(items).most_common(1)[0][0]


def current_game_state():
    """Get current game state"""
    cards = []

    for card_id in track_last_seen:
        if frame_count - track_last_seen[card_id] > MAX_MISSING_FRAMES:
            continue

        labels = track_label_history[card_id]
        if len(labels) < MIN_STABLE_FRAMES:
            continue

        label = majority_list(labels)
        if label:
            cards.append(format_card(label))

    print(f"raw cards: {cards}")
    return sorted(set(cards))


while True:
    """OpenCV Loop"""
    plugged, frame = cap.read()

    if not plugged:
        break

    results = model.track(frame, persist=True, conf=MIN_CONF, verbose=False)

    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        ids = boxes.id
        clss = boxes.cls
        confs = boxes.conf

        if ids is not None:
            for track_id_tensor, cls_tensor, conf_tensor in zip(ids, clss, confs):
                track_id = int(track_id_tensor.item())
                cls_id = int(cls_tensor.item())
                conf = float(conf_tensor.item())

                if conf < MIN_CONF:
                    continue

                label = model.names[cls_id]
                track_label_history[track_id].append(label)
                track_last_seen[track_id] = frame_count
                track_label_history[track_id] = track_label_history[track_id][-20:]

    cards = current_game_state()

    annotated = results[0].plot() if results else frame.copy()

    cv2.putText(annotated, f"Cards: {cards}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)

    cv2.imshow("Card Tracker", annotated)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()