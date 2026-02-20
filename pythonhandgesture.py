import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Screen size
screen_w, screen_h = pyautogui.size()

# Camera
cap = cv2.VideoCapture(0)

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils

click = False
right_click = False
scroll_time = 0


# Finger status
def fingers_up(lm):
    fingers = []

    # Thumb
    fingers.append(lm[4].x < lm[3].x)

    # Other fingers
    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(lm[tip].y < lm[tip-2].y)

    return fingers


while True:

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            lm = hand.landmark

            finger = fingers_up(lm)
            count = finger.count(True)

            # Index finger
            x = int(lm[8].x * w)
            y = int(lm[8].y * h)

            screen_x = screen_w / w * x
            screen_y = screen_h / h * y

            # Move mouse (1 finger)
            if count == 1:
                pyautogui.moveTo(screen_x, screen_y, duration=0.05)

            # Left click (Pinch)
            x2 = int(lm[4].x * w)
            y2 = int(lm[4].y * h)

            dist = math.hypot(x2-x, y2-y)

            if dist < 30:
                if not click:
                    pyautogui.click()
                    click = True
            else:
                click = False

            # Scroll (2 fingers)
            if count == 2:
                if time.time() - scroll_time > 0.3:
                    pyautogui.scroll(200)
                    scroll_time = time.time()

            # Right click (3 fingers)
            if count == 3:
                if not right_click:
                    pyautogui.rightClick()
                    right_click = True
            else:
                right_click = False

            # Pause (Fist)
            if count == 0:
                cv2.putText(frame, "PAUSED", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,0,255), 3)

            cv2.circle(frame, (x, y), 10, (0,255,0), -1)

    cv2.imshow("Hand Control Pro", frame)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
