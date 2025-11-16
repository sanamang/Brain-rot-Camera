import cv2
import mediapipe as mp
import random      # for random window positions
import time        # for timing between gestures
import math        # for distance between landmarks

# --- Config ---
MOTION_HISTORY = 4          # number of frames to look back (shorter = more responsive)
MOTION_THRESHOLD_Y = 0.004  # vertical movement sensitivity (6–7) - more sensitive

IMAGE_PATH_67 = "meme.jpeg"      # image to show on 6–7 gesture
IMAGE_PATH_CLOCK = "meme2.png"   # image to show on clock-it gesture

CLOCK_IT_BLOCK_AFTER_67 = 1.0    # seconds to block clock-it after a 6–7
PINCH_THRESHOLD = 0.05           # distance threshold (normalized) for thumb-index pinch
OPEN_FINGER_THRESHOLD = 0.10     # min distance from palm to each fingertip to count as "open"

# --- Setup camera + mediapipe ---
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

meme_67 = cv2.imread(IMAGE_PATH_67)
if meme_67 is None:
    raise FileNotFoundError(f"Could not load {IMAGE_PATH_67}")

meme_clock = cv2.imread(IMAGE_PATH_CLOCK)
if meme_clock is None:
    raise FileNotFoundError(f"Could not load {IMAGE_PATH_CLOCK}")

# store last few Y positions for each hand (0 and 1) for 6–7
history_y = {0: [], 1: []}

trigger_count_67 = 0
trigger_count_clock = 0

last_67_time = 0.0  # last time a 6–7 gesture was detected

# track previous pinch state so we only trigger on "pinch start"
pinch_prev = {0: False, 1: False}


def get_direction(hist):
    """
    Rough direction of motion:
    > 0  -> moving in + direction
    < 0  -> moving in - direction
    """
    if len(hist) < 2:
        return 0.0
    return hist[-1] - hist[0]


def is_hand_open(palm, finger_tips):
    """
    Determine if a hand is "open":
    All four fingertips (index, middle, ring, pinky) must be
    far enough from the palm center.
    """
    px, py = palm
    dists = []
    for fx, fy in finger_tips:
        d = math.sqrt((fx - px) ** 2 + (fy - py) ** 2)
        dists.append(d)
    return all(d > OPEN_FINGER_THRESHOLD for d in dists)


# CAMERA WINDOW LOCATION (adjust if your webcam resolution is different)
CAMERA_X = 50
CAMERA_Y = 50
CAMERA_W = 640     # width of your camera window
CAMERA_H = 480     # height of your camera window

# POPUPS WILL BE PLACED TO THE RIGHT OF THE CAMERA
RIGHT_START_X = CAMERA_X + CAMERA_W + 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # per-frame data
    current_hand_positions = {}  # idx -> (x, y) for wrist
    thumb_tips = {}              # idx -> (x, y)
    index_tips = {}              # idx -> (x, y)
    palm_centers = {}            # idx -> (x, y) for palm
    finger_tips_all = {}         # idx -> list of (x, y) for index/middle/ring/pinky

    if result.multi_hand_landmarks and result.multi_handedness:
        for idx, (handLms, handedness) in enumerate(
            zip(result.multi_hand_landmarks, result.multi_handedness)
        ):
            # Wrist (landmark 0) as palm proxy, or use landmark 9 (middle of palm)
            palm = handLms.landmark[0]  # you could also try 9
            x_palm = palm.x
            y_palm = palm.y
            palm_centers[idx] = (x_palm, y_palm)

            # Wrist position (for vertical tracking)
            current_hand_positions[idx] = (x_palm, y_palm)

            # Thumb tip (landmark 4) and index finger tip (landmark 8) for clock-it pinch
            thumb = handLms.landmark[4]
            index = handLms.landmark[8]
            thumb_tips[idx] = (thumb.x, thumb.y)
            index_tips[idx] = (index.x, index.y)

            # Finger tips for "open hand" check: index(8), middle(12), ring(16), pinky(20)
            tips = []
            for lm_id in [8, 12, 16, 20]:
                lm = handLms.landmark[lm_id]
                tips.append((lm.x, lm.y))
            finger_tips_all[idx] = tips

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame, handLms, mp_hands.HAND_CONNECTIONS
            )

            label = handedness.classification[0].label  # "Left" or "Right"
            cv2.putText(
                frame, label,
                (int(x_palm * w), int(y_palm * h)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

    # update motion history for 6–7 (vertical)
    for idx in [0, 1]:
        if idx in current_hand_positions:
            _, y = current_hand_positions[idx]
            history_y[idx].append(y)
            if len(history_y[idx]) > MOTION_HISTORY:
                history_y[idx].pop(0)
        else:
            history_y[idx].clear()
            pinch_prev[idx] = False  # reset pinch state when hand disappears

    # ---------- Gesture detection ----------
    gesture_67 = False
    gesture_clock = False

    num_hands = len(current_hand_positions)
    now = time.time()

    # 6–7 gesture: both hands, opposite vertical motion AND both hands open
    if (
        num_hands == 2
        and len(history_y[0]) >= 2
        and len(history_y[1]) >= 2
        and 0 in palm_centers and 1 in palm_centers
        and 0 in finger_tips_all and 1 in finger_tips_all
    ):
        dy0 = get_direction(history_y[0])
        dy1 = get_direction(history_y[1])

        moving0 = abs(dy0) > (MOTION_THRESHOLD_Y * 0.5)
        moving1 = abs(dy1) > (MOTION_THRESHOLD_Y * 0.5)
        opposite = dy0 * dy1 < 0  # one up, one down

        hand0_open = is_hand_open(palm_centers[0], finger_tips_all[0])
        hand1_open = is_hand_open(palm_centers[1], finger_tips_all[1])

        if moving0 and moving1 and opposite and hand0_open and hand1_open:
            gesture_67 = True
            last_67_time = now
            print("6–7 gesture detected with BOTH hands open!")
            history_y[0].clear()
            history_y[1].clear()
            pinch_prev[0] = False
            pinch_prev[1] = False

    # clock-it gesture: thumb + index finger touching on EXACTLY ONE hand
    if (not gesture_67) and (now - last_67_time > CLOCK_IT_BLOCK_AFTER_67) and num_hands == 1:
        idx = list(current_hand_positions.keys())[0]  # the visible hand
        if idx in thumb_tips and idx in index_tips:
            tx, ty = thumb_tips[idx]
            ix, iy = index_tips[idx]
            dist = math.sqrt((tx - ix) ** 2 + (ty - iy) ** 2)

            is_pinch = dist < PINCH_THRESHOLD

            if is_pinch and not pinch_prev[idx]:
                gesture_clock = True
                print(f"clock-it pinch detected with hand {idx}! (dist={dist:.4f})")

            pinch_prev[idx] = is_pinch

    # ---------- Show memes ----------

    if gesture_67:
        trigger_count_67 += 1
        window_name = f"6 7 MEME #{trigger_count_67}"
        resized = cv2.resize(meme_67, (350, 350))
        cv2.imshow(window_name, resized)

        # Safe random spawn area on RIGHT side only
        x_win = random.randint(RIGHT_START_X, RIGHT_START_X + 600)
        y_win = random.randint(50, 700)
        cv2.moveWindow(window_name, x_win, y_win)

    if gesture_clock:
        trigger_count_clock += 1
        window_name = f"CLOCK IT MEME #{trigger_count_clock}"
        resized = cv2.resize(meme_clock, (350, 350))
        cv2.imshow(window_name, resized)

        # Safe random spawn area on RIGHT side only
        x_win = random.randint(RIGHT_START_X, RIGHT_START_X + 600)
        y_win = random.randint(50, 700)
        cv2.moveWindow(window_name, x_win, y_win)

    # show camera feed
    cv2.imshow("Gesture Detector", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
