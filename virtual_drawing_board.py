import cv2
import mediapipe as mp
import numpy as np

# =========================
# MediaPipe Setup
# =========================
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

print("Drawing Board Ready!")

# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(0)

# =========================
# Canvas
# =========================
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

prev_x = None
prev_y = None

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        index_tip = hand[8]

        h, w, _ = frame.shape

        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        # Draw fingertip marker
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

        # Draw on canvas
        if prev_x is not None and prev_y is not None:

            cv2.line(
                canvas,
                (prev_x, prev_y),
                (x, y),
                (0, 255, 0),
                5
            )

        prev_x = x
        prev_y = y

    else:
        prev_x = None
        prev_y = None

    # Combine drawing canvas with webcam frame
    output = cv2.add(frame, canvas)

    cv2.putText(
        output,
        "Press C = Clear | ESC = Exit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Virtual Drawing Board", output)

    key = cv2.waitKey(1) & 0xFF

    # Clear Canvas
    if key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()