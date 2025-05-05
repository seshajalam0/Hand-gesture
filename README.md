
**Real-Time Hand Gesture Recognition for YouTube Control Using CNN-LSTM and MediaPipe
**

**1. Problem Statement**  
The goal of the project was to build a real-time hand gesture recognition system that could detect and classify specific hand gestures and map them to YouTube playback controls (e.g., play, pause, forward, rewind) using webcam input.


**2. Dataset Collection and Preprocessing**  
- The **Hand Gesture Recognition Database** was used, consisting of multiple gesture classes recorded under different lighting and angles.  
- **MediaPipe Hands** was used to extract **21 3D landmarks** (x, y, z) per hand from each frame of a gesture sequence.  
- For each gesture sample, sequences of landmark coordinates were stored as `.npy` files, where each sequence represented a gesture instance over time.  
- A **fixed-length window** (e.g., 30 frames) was used to maintain temporal consistency for each gesture.


**3. Feature Extraction**  
- Each video frame was processed using **MediaPipe**, which accurately detects the hand and extracts 21 landmarks.
- Each landmark contains 3 values (x, y, z), resulting in **63 features per frame**.
- The time-series data was stored as a sequence of shape `(30, 63)` for each gesture.


**4. Model Architecture – CNN-LSTM**  
  - The model consisted of:
  - A **1D Convolutional layer** to capture spatial patterns in the landmarks across the time window.
  - **LSTM layers** to learn temporal dependencies between frames.
  - **Dense layers** for classification of gesture classes.
  - The final output layer used **Softmax activation** to output probabilities over all gesture classes.


**5. Training the Model ** 
  - The model was trained using TensorFlow with:
  - **Categorical cross-entropy loss**
  - **Adam optimizer**
  - **EarlyStopping and model checkpointing** to prevent overfitting.
  - Training data was split into training and validation sets.
  - Achieved high accuracy on validation data (you can mention the exact percentage here).


**6. Real-Time Prediction Pipeline ** 
- Integrated the trained model into a **real-time OpenCV loop** to capture frames from the webcam.
- Each frame was passed through **MediaPipe Hands** to extract landmarks.
- A buffer of the last 30 frames was maintained to create live sequences for prediction.
- The model predicted gestures in real time and applied smoothing to avoid flickering.


**7. Action Mapping with PyAutoGUI ** 
  - Recognized gestures were mapped to specific **YouTube actions**:
  - Play/Pause
  - Rewind
  - Fast Forward
  - Volume Up/Down
  - Used PyAutoGUI to simulate keyboard shortcuts for controlling playback.


**8. Evaluation & Results ** 
- Evaluated model using accuracy, confusion matrix, and classification report.
- Real-time performance was tested under various lighting and hand positions to ensure robustness.
- The model showed reliable gesture recognition with minimal latency.


**9. Outcome **

The system successfully recognized multiple hand gestures in real time and controlled YouTube playback without physical input devices. The project demonstrated practical integration of computer vision, deep learning, and user interaction using Python, MediaPipe, TensorFlow, and OpenCV.

