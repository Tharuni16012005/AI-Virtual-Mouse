import cv2
import mediapipe as mp
import pyautogui
import math

# =========================
# Screen Setup
# =========================
screen_width, screen_height = pyautogui.size()

pyautogui.FAILSAFE = False

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

print("Hand Landmarker Loaded!")

# =========================
# Webcam
# =========================
cap = cv2.VideoCapture(0)

# =========================
# Mouse Smoothing
# =========================
prev_x = 0
prev_y = 0
smoothening = 5

# =========================
# Click Protection
# =========================
left_click_done = False
right_click_done = False
double_click_done = False

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

        # Landmark points
        thumb_tip = hand[4]
        index_tip = hand[8]
        middle_tip = hand[12]
        ring_tip = hand[16]

        h, w, _ = frame.shape

        # Draw landmarks
        thumb_x = int(thumb_tip.x * w)
        thumb_y = int(thumb_tip.y * h)

        index_x = int(index_tip.x * w)
        index_y = int(index_tip.y * h)

        middle_x = int(middle_tip.x * w)
        middle_y = int(middle_tip.y * h)

        ring_x = int(ring_tip.x * w)
        ring_y = int(ring_tip.y * h)

        cv2.circle(frame, (thumb_x, thumb_y), 12, (255, 0, 0), -1)
        cv2.circle(frame, (index_x, index_y), 12, (0, 0, 255), -1)
        cv2.circle(frame, (middle_x, middle_y), 12, (0, 255, 0), -1)
        cv2.circle(frame, (ring_x, ring_y), 12, (0, 255, 255), -1)

        # =========================
        # Cursor Movement
        # =========================

        target_x = index_tip.x * screen_width
        target_y = index_tip.y * screen_height

        curr_x = prev_x + (target_x - prev_x) / smoothening
        curr_y = prev_y + (target_y - prev_y) / smoothening

        pyautogui.moveTo(curr_x, curr_y)

        prev_x = curr_x
        prev_y = curr_y

        # =========================
        # Distances
        # =========================

        left_distance = math.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 +
            (thumb_tip.y - index_tip.y) ** 2
        )

        right_distance = math.sqrt(
            (thumb_tip.x - middle_tip.x) ** 2 +
            (thumb_tip.y - middle_tip.y) ** 2
        )

        double_distance = math.sqrt(
            (thumb_tip.x - ring_tip.x) ** 2 +
            (thumb_tip.y - ring_tip.y) ** 2
        )

        # Display distances
        cv2.putText(
            frame,
            f"L:{left_distance:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"R:{right_distance:.2f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"D:{double_distance:.2f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # =========================
        # LEFT CLICK
        # =========================

        if left_distance < 0.20:

            cv2.putText(
                frame,
                "LEFT CLICK",
                (250, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

            if not left_click_done:
                pyautogui.click()
                left_click_done = True

        else:
            left_click_done = False

        # =========================
        # RIGHT CLICK
        # =========================

        if right_distance < 0.20:

            cv2.putText(
                frame,
                "RIGHT CLICK",
                (250, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                3
            )

            if not right_click_done:
                pyautogui.rightClick()
                right_click_done = True

        else:
            right_click_done = False

        # =========================
        # DOUBLE CLICK
        # =========================

        if double_distance < 0.20:

            cv2.putText(
                frame,
                "DOUBLE CLICK",
                (250, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                3
            )

            if not double_click_done:
                pyautogui.doubleClick()
                double_click_done = True

        else:
            double_click_done = False

    cv2.imshow("AI Virtual Mouse PRO", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()