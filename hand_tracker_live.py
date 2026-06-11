import cv2
import mediapipe as mp

# ==========================
# MediaPipe Setup
# ==========================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

print("Loading Hand Landmarker...")

detector = HandLandmarker.create_from_options(options)

print("Hand Landmarker Loaded Successfully!")
print("Starting Webcam...")

# ==========================
# Open Webcam
# ==========================

cap = cv2.VideoCapture(0)

print("Webcam Opened:", cap.isOpened())

# ==========================
# Main Loop
# ==========================

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read webcam frame")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hand
    result = detector.detect(mp_image)

    # If hand detected
    if result.hand_landmarks:

        cv2.putText(
            frame,
            "Hand Detected!",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        hand = result.hand_landmarks[0]

        # Landmark 8 = Index Finger Tip
        index_tip = hand[8]

        h, w, _ = frame.shape

        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        # Draw large red dot
        cv2.circle(frame, (x, y), 25, (0, 0, 255), -1)

        # Show coordinates
        cv2.putText(
            frame,
            f"X:{x} Y:{y}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        print(f"Index Finger: ({x}, {y})")

    # Show webcam window
    cv2.imshow("Hand Tracker", frame)

    # ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================
# Cleanup
# ==========================

cap.release()
cv2.destroyAllWindows()