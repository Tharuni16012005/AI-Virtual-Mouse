import cv2
import mediapipe as mp
import math
from pycaw.pycaw import AudioUtilities

# =========================
# Windows Volume Setup
# =========================

speakers = AudioUtilities.GetSpeakers()
volume = speakers.EndpointVolume

print("Volume Control Ready!")

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

detector = HandLandmarker.create_from_options(options)

# =========================
# Webcam
# =========================

cap = cv2.VideoCapture(0)

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

        thumb_tip = hand[4]
        index_tip = hand[8]

        h, w, _ = frame.shape

        thumb_x = int(thumb_tip.x * w)
        thumb_y = int(thumb_tip.y * h)

        index_x = int(index_tip.x * w)
        index_y = int(index_tip.y * h)

        # Draw circles
        cv2.circle(frame, (thumb_x, thumb_y), 15, (255, 0, 0), -1)
        cv2.circle(frame, (index_x, index_y), 15, (0, 0, 255), -1)

        # Draw line
        cv2.line(
            frame,
            (thumb_x, thumb_y),
            (index_x, index_y),
            (0, 255, 0),
            3
        )

        # Distance between fingers
        distance = math.sqrt(
            (thumb_x - index_x) ** 2 +
            (thumb_y - index_y) ** 2
        )

        # Convert distance to volume percentage
        volume_level = distance / 300

        volume_level = max(
            0.0,
            min(1.0, volume_level)
        )

        # Set Windows volume
        volume.SetMasterVolumeLevelScalar(
            volume_level,
            None
        )

        volume_percent = int(volume_level * 100)

        # Volume bar
        bar_height = int(
            (volume_percent / 100) * 300
        )

        cv2.rectangle(
            frame,
            (50, 100),
            (100, 400),
            (255, 255, 255),
            2
        )

        cv2.rectangle(
            frame,
            (50, 400 - bar_height),
            (100, 400),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            f"{volume_percent}%",
            (30, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Distance: {int(distance)}",
            (150, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        "AI Volume Control",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()