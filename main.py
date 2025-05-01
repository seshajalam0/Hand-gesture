import cv2
import mediapipe as mp
import pyautogui
import time
import math


def count_fingers(lst):
    """Count the number of fingers raised using hand landmarks."""
    cnt = 0
    thresh = abs((lst.landmark[0].y - lst.landmark[9].y) * 100) / 2

    # Check four fingers (index, middle, ring, pinky)
    for tip, base in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        if (lst.landmark[base].y - lst.landmark[tip].y) * 100 > thresh:
            cnt += 1

    # Check thumb (based on angle for better accuracy)
    thumb_tip = lst.landmark[4]
    thumb_base = lst.landmark[2]
    index_base = lst.landmark[5]

    angle = math.degrees(math.atan2(thumb_tip.y - thumb_base.y, thumb_tip.x - thumb_base.x) -
                         math.atan2(index_base.y - thumb_base.y, index_base.x - thumb_base.x))
    angle = abs(angle)
    if angle > 40:  # Thumb is raised
        cnt += 1

    return cnt, angle


# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand_obj = mp_hands.Hands(max_num_hands=2)

prev = [-1, -1]
start_init = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_obj.process(rgb_frame)

    hand_counts = []
    thumb_angles = []

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            count, thumb_angle = count_fingers(hand_landmarks)
            hand_counts.append(count)
            thumb_angles.append(thumb_angle)

            if prev[i] != count:
                if not start_init:
                    start_time = time.time()
                    start_init = True
                elif time.time() - start_time > 0.2:
                    key_map = {
                        0: 'space',  # Play/Pause using fist (0 fingers)
                        1: 'right',  # Next video
                        2: 'left',  # Previous video
                        3: 'volumeup',  # Increase volume
                        4: 'volumedown',  # Decrease volume
                        5: 'f',  # Fullscreen
                        6: 'esc',  # Exit Fullscreen
                        7: 'm',  # Mute/Unmute
                        8: 'up',  # Increase speed
                        9: 'down'  # Decrease speed
                    }
                    if count in key_map:
                        pyautogui.press(key_map[count])

                    # Updated gestures for Like and Dislike
                    if count == 1:  # Thumbs up for Like
                        pyautogui.press('l')  # Like video
                    elif count == 2:  # Thumbs down for Dislike
                        pyautogui.press('d')  # Dislike video

                    prev[i] = count
                    start_init = False

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Two-hand gestures
        if len(hand_counts) == 2:
            total_fingers = hand_counts[0] + hand_counts[1]
            if total_fingers == 8:
                pyautogui.press('up')  # Increase playback speed
            elif total_fingers == 9:
                pyautogui.press('down')  # Decrease playback speed
            elif total_fingers == 6:
                pyautogui.press('esc')  # Exit fullscreen
            elif total_fingers == 5:
                pyautogui.press('m')  # Mute/Unmute

    else:
        cv2.putText(frame, "No Hand Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Left Hand: {prev[0]}, Right Hand: {prev[1]}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
