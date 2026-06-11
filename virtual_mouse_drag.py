import cv2
import mediapipe as mp
import pyautogui
import math
import time

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
# Gesture Flags
# =========================
left_click_done = False

dragging = False
pinch_start_time = 0

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

        # =========================
        # Draw Landmarks
        # =========================
        thumb_x = int(thumb_tip.x * w)
        thumb_y = int(thumb_tip.y * h)

        index_x = int(index_tip.x * w)
        index_y = int(index_tip.y * h)

        cv2.circle(frame, (thumb_x, thumb_y), 15, (255, 0, 0), -1)
        cv2.circle(frame, (index_x, index_y), 15, (0, 0, 255), -1)

        # =========================
        # Mouse Movement
        # =========================
        target_x = index_tip.x * screen_width
        target_y = index_tip.y * screen_height

        curr_x = prev_x + (target_x - prev_x) / smoothening
        curr_y = prev_y + (target_y - prev_y) / smoothening

        pyautogui.moveTo(curr_x, curr_y)

        prev_x = curr_x
        prev_y = curr_y

        # =========================
        # Distance
        # =========================
        distance = math.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 +
            (thumb_tip.y - index_tip.y) ** 2
        )

        cv2.putText(
            frame,
            f"Distance: {distance:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # =========================
        # Pinch Detection
        # =========================
        if distance < 0.20:

            if pinch_start_time == 0:
                pinch_start_time = time.time()

            pinch_duration = time.time() - pinch_start_time

            cv2.putText(
                frame,
                f"Hold: {pinch_duration:.1f}s",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # Quick pinch = Left Click
            if pinch_duration < 1.0:

                cv2.putText(
                    frame,
                    "LEFT CLICK",
                    (250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                if not left_click_done:
                    pyautogui.mouseDown()
                    left_click_done = True

            # Long pinch = Drag Start
            else:

                cv2.putText(
                    frame,
                    "DRAGGING",
                    (250, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                if not dragging:
                    print("MOUSE DOWN")
                    pyautogui.mouseDown()
                    dragging = True
                    
                    

        else:

            pinch_start_time = 0
            left_click_done = False

            if dragging:
             print("MOUSE UP")
             pyautogui.mouseUp()
             dragging = False
            

    cv2.imshow("AI Virtual Mouse Drag", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()