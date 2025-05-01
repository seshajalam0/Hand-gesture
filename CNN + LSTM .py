
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import tensorflow as tf
from collections import deque

# Load the trained CNN + LSTM model
model = tf.keras.models.load_model(r"C:\Users\seshajalam\Desktop\DA\projects\hand gesture\cnn_lstm_gesture_model.h5")

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand_obj = mp_hands.Hands(max_num_hands=1)

# Define action mapping based on model output
gesture_map = {
    0: 'space',  # Play/Pause
    1: 'right',  # Next video
    2: 'left',  # Previous video
    3: 'volumeup',  # Increase volume
    4: 'volumedown',  # Decrease volume
    5: 'f',  # Fullscreen
    6: 'esc',  # Exit fullscreen
    7: 'm',  # Mute
    8: 'up',  # Increase speed
    9: 'down'  # Decrease speed
}

# Store last N frames for LSTM input
sequence_length = 10
landmark_queue = deque(maxlen=sequence_length)

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_obj.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract 21 landmarks (x, y, z) and flatten them
            landmark_data = []
            for lm in hand_landmarks.landmark:
                landmark_data.extend([lm.x, lm.y, lm.z])

            # Append to queue
            landmark_queue.append(landmark_data)

            # Predict gesture if enough frames are stored
            if len(landmark_queue) == sequence_length:
                input_data = np.expand_dims(landmark_queue, axis=0)  # Shape (1, 10, 63)
                prediction = model.predict(input_data)
                gesture_index = np.argmax(prediction)

                if gesture_index in gesture_map:
                    pyautogui.press(gesture_map[gesture_index])
                    cv2.putText(frame, f"Action: {gesture_map[gesture_index]}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv2.putText(frame, "No Hand Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
