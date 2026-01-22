import cv2
import mediapipe as mp
import joblib
from collections import deque

model = joblib.load("model/sign_model.pkl")

GESTURE_TO_TEXT = {
    "hi": "Hi",
    "how": "How",
    "are_you": "are you?",
    "fine": "I am fine.",
    "What ": "What ",
    "is your name ":"is your name",
    "my": "My ",
    "name is ": "name is sachin"
}

def put_text_next_line(img, text, x, y, max_chars=30, line_gap=35):
    words = text.split(" ")
    line = ""
    row = 0

    for word in words:
        if len(line + word) <= max_chars:
            line += word + " "
        else:
            cv2.putText(img, line.strip(),
                        (x, y + row * line_gap),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            row += 1
            line = word + " "

    if line:
        cv2.putText(img, line.strip(),
                    (x, y + row * line_gap),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

gesture_buffer = deque(maxlen=15)
sentence = []

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            data = []
            for lm in hand.landmark:
                data.extend([lm.x, lm.y, lm.z])

            pred = model.predict([data])[0]
            gesture_buffer.append(pred)

            if gesture_buffer.count(pred) > 10:
                if not sentence or sentence[-1] != pred:
                    sentence.append(pred)

    display_sentence = " ".join(GESTURE_TO_TEXT.get(w, w) for w in sentence)

    put_text_next_line(frame, display_sentence, 10, 40)

    cv2.imshow("Sign Language to Sentence", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
