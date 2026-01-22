import cv2
import mediapipe as mp
import csv
import os

LABEL = "is your name  "   # 🔁 change label every time
SAVE_PATH = f"data/{LABEL}.csv"

os.makedirs("data", exist_ok=True)

file = open(SAVE_PATH, "a", newline="")
writer = csv.writer(file)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

print(f"Collecting data for gesture: {LABEL}")
print("Press ESC to stop")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            row = []
            for lm in hand.landmark:
                row.extend([lm.x, lm.y, lm.z])
            row.append(LABEL)
            writer.writerow(row)

    cv2.imshow("Data Collection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
file.close()
cv2.destroyAllWindows()
