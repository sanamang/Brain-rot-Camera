import cv2
import mediapipe as mp
import random
import time
import math
import imageio

MOTION_HISTORY = 4
MOTION_THRESHOLD_Y = 0.004

IMAGE_PATH_67 = "meme.jpeg"           
IMAGE_PATH_CLOCK = "meme2.png"        
GIF_PATH = "rizzler-the-rizzle.gif"    
IMAGE_PATH_FIST_CHIN = "ragebait-rage.png" 

CLOCK_IT_BLOCK_AFTER_67 = 1.0
PINCH_THRESHOLD = 0.03        
MIN_UNPINCH_DIST = 0.07       
OPEN_FINGER_THRESHOLD = 0.10

SHUSH_LIPS_START_DIST = 0.05   
SHUSH_COOLDOWN = 1.0           

FIST_FINGER_CLOSE = 0.12       
FIST_CHIN_DIST = 0.12         
FIST_COOLDOWN = 1.0            

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=4,   
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

meme_67 = cv2.imread(IMAGE_PATH_67)
if meme_67 is None:
    raise FileNotFoundError(f"Could not load {IMAGE_PATH_67}")

meme_clock = cv2.imread(IMAGE_PATH_CLOCK)
if meme_clock is None:
    raise FileNotFoundError(f"Could not load {IMAGE_PATH_CLOCK}")

meme_fist = cv2.imread(IMAGE_PATH_FIST_CHIN)
if meme_fist is None:
    raise FileNotFoundError(f"Could not load {IMAGE_PATH_FIST_CHIN}")


gif_reader = imageio.get_reader(GIF_PATH)
gif_frames = []
for frame in gif_reader:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(frame_bgr, (350, 350))
    gif_frames.append(resized)


history_y = {}          
pinch_prev = {}        
prev_pinch_dist = {}    

trigger_count_67 = 0
trigger_count_clock = 0
trigger_count_shush = 0
trigger_count_fist = 0

last_67_time = 0.0
last_shush_time = 0.0
last_fist_time = 0.0


gif_active = False
gif_index = 0
gif_window_name = ""
GIF_FRAME_INTERVAL = 0.03   
gif_last_time = 0.0

CAMERA_X = 50
CAMERA_Y = 50
CAMERA_W = 640
CAMERA_H = 480
RIGHT_START_X = CAMERA_X + CAMERA_W + 30
POPUP_MAX_X_OFFSET = 400   
POPUP_MIN_Y = 80          
POPUP_MAX_Y = 380         


def ensure_hand_state(idx):
    """Make sure all per-hand dicts have entries for this index."""
    if idx not in history_y:
        history_y[idx] = []
    if idx not in pinch_prev:
        pinch_prev[idx] = False
    if idx not in prev_pinch_dist:
        prev_pinch_dist[idx] = 1.0


def get_direction(hist):
    if len(hist) < 2:
        return 0.0
    return hist[-1] - hist[0]


def is_hand_open(palm, finger_tips):
    px, py = palm
    for fx, fy in finger_tips:
        d = math.sqrt((fx - px) ** 2 + (fy - py) ** 2)
        if d <= OPEN_FINGER_THRESHOLD:
            return False
    return True


def is_hand_open_upward(palm, finger_tips):
    """
    Open *and* fingers are above the palm (palms 'up' / hands raised).
    """
    if not is_hand_open(palm, finger_tips):
        return False

    px, py = palm
    ys = [fy for (_, fy) in finger_tips]
    avg_y = sum(ys) / len(ys)
    
    return avg_y < py - 0.02


def dist2(a, b):
    ax, ay = a
    bx, by = b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def is_fist(palm, finger_tips):
    """
    Fist: all fingertips relatively close to the palm.
    """
    px, py = palm
    for fx, fy in finger_tips:
        d = math.sqrt((fx - px) ** 2 + (fy - py) ** 2)
        if d > FIST_FINGER_CLOSE:
            return False
    return True


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(frame_rgb)
    face_results = face_mesh.process(frame_rgb)

    lips_point = None
    chin_point = None
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
      
        lm_lips = face_landmarks.landmark[13]
        lm_chin = face_landmarks.landmark[152]
        lips_point = (lm_lips.x, lm_lips.y)
        chin_point = (lm_chin.x, lm_chin.y)

        lips_x, lips_y = lips_point
        chin_x, chin_y = chin_point

        
        cv2.circle(frame, (int(lips_x * w), int(lips_y * h)), 4, (0, 255, 0), -1)
        cv2.circle(frame, (int(chin_x * w), int(chin_y * h)), 4, (255, 0, 0), -1)

   
    current_hand_positions = {}
    thumb_tips = {}
    index_tips = {}
    palm_centers = {}
    finger_tips_all = {}

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for idx, (handLms, handedness) in enumerate(
            zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)
        ):
            ensure_hand_state(idx)

            palm = handLms.landmark[0]
            x_palm = palm.x
            y_palm = palm.y
            palm_centers[idx] = (x_palm, y_palm)
            current_hand_positions[idx] = (x_palm, y_palm)

            thumb = handLms.landmark[4]
            index = handLms.landmark[8]
            thumb_tips[idx] = (thumb.x, thumb.y)
            index_tips[idx] = (index.x, index.y)

            tips = []
            for lm_id in [8, 12, 16, 20]:
                lm = handLms.landmark[lm_id]
                tips.append((lm.x, lm.y))
            finger_tips_all[idx] = tips

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            label = handedness.classification[0].label
            cv2.putText(
                frame, label,
                (int(x_palm * w), int(y_palm * h)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

    for idx in list(history_y.keys()):
        if idx in current_hand_positions:
            _, y = current_hand_positions[idx]
            history_y[idx].append(y)
            if len(history_y[idx]) > MOTION_HISTORY:
                history_y[idx].pop(0)
        else:
            history_y[idx].clear()
            pinch_prev[idx] = False
            prev_pinch_dist[idx] = 1.0

    gesture_67 = False
    gesture_clock = False
    gesture_shush = False
    gesture_fistchin = False

    num_hands = len(current_hand_positions)
    now = time.time()

    if (
        num_hands == 2
        and 0 in current_hand_positions
        and 1 in current_hand_positions
        and len(history_y.get(0, [])) >= 2
        and len(history_y.get(1, [])) >= 2
        and 0 in palm_centers and 1 in palm_centers
        and 0 in finger_tips_all and 1 in finger_tips_all
    ):
        dy0 = get_direction(history_y[0])
        dy1 = get_direction(history_y[1])

        moving0 = abs(dy0) > (MOTION_THRESHOLD_Y * 0.5)
        moving1 = abs(dy1) > (MOTION_THRESHOLD_Y * 0.5)
        opposite = dy0 * dy1 < 0  

        hand0_ok = is_hand_open_upward(palm_centers[0], finger_tips_all[0])
        hand1_ok = is_hand_open_upward(palm_centers[1], finger_tips_all[1])

        if moving0 and moving1 and opposite and hand0_ok and hand1_ok:
            gesture_67 = True
            last_67_time = now
            print("6–7 gesture detected (palms open & up)!")
            history_y[0].clear()
            history_y[1].clear()
            pinch_prev[0] = False
            pinch_prev[1] = False
            prev_pinch_dist[0] = 1.0
            prev_pinch_dist[1] = 1.0

    if (not gesture_67) and (now - last_67_time > CLOCK_IT_BLOCK_AFTER_67) and num_hands >= 1:
        for idx in current_hand_positions.keys():
            ensure_hand_state(idx)
            if idx in thumb_tips and idx in index_tips:
                tx, ty = thumb_tips[idx]
                ix, iy = index_tips[idx]
                dist_pinch = math.sqrt((tx - ix)**2 + (ty - iy)**2)

                is_pinch_now = dist_pinch < PINCH_THRESHOLD
                clearly_unpinched_before = prev_pinch_dist[idx] > MIN_UNPINCH_DIST

                if is_pinch_now and clearly_unpinched_before and not pinch_prev[idx]:
                    gesture_clock = True
                    print(f"clock-it pinch detected (hand {idx})")

                pinch_prev[idx] = is_pinch_now
                prev_pinch_dist[idx] = dist_pinch

    if lips_point is not None:
        for idx in current_hand_positions.keys():
            if idx in index_tips:
                finger_pos = index_tips[idx]
                d_lips = dist2(finger_pos, lips_point)

                if d_lips < SHUSH_LIPS_START_DIST and (now - last_shush_time) > SHUSH_COOLDOWN:
                    gesture_shush = True
                    last_shush_time = now
                    print("Shush gesture detected (finger on lips).")
                    break  


    if chin_point is not None:
        for idx in current_hand_positions.keys():
            if idx in palm_centers and idx in finger_tips_all:
                palm = palm_centers[idx]
                tips = finger_tips_all[idx]

                if is_fist(palm, tips):
                    d_chin = dist2(palm, chin_point)
                    if d_chin < FIST_CHIN_DIST and (now - last_fist_time) > FIST_COOLDOWN:
                        gesture_fistchin = True
                        last_fist_time = now
                        print("Fist-under-chin gesture detected!")
                        break

    if gesture_67:
        trigger_count_67 += 1
        window_name = f"6 7 MEME #{trigger_count_67}"
        cv2.imshow(window_name, cv2.resize(meme_67, (350, 350)))
        cv2.moveWindow(
            window_name,
            random.randint(RIGHT_START_X, RIGHT_START_X + POPUP_MAX_X_OFFSET),
            random.randint(POPUP_MIN_Y, POPUP_MAX_Y),
        )

    if gesture_clock:
        trigger_count_clock += 1
        window_name = f"CLOCK IT MEME #{trigger_count_clock}"
        cv2.imshow(window_name, cv2.resize(meme_clock, (350, 350)))
        cv2.moveWindow(
            window_name,
            random.randint(RIGHT_START_X, RIGHT_START_X + POPUP_MAX_X_OFFSET),
            random.randint(POPUP_MIN_Y, POPUP_MAX_Y),
        )

    if gesture_fistchin:
        trigger_count_fist += 1
        window_name = f"FIST CHIN MEME #{trigger_count_fist}"
        cv2.imshow(window_name, cv2.resize(meme_fist, (350, 350)))
        cv2.moveWindow(
            window_name,
            random.randint(RIGHT_START_X, RIGHT_START_X + POPUP_MAX_X_OFFSET),
            random.randint(POPUP_MIN_Y, POPUP_MAX_Y),
        )

    if gesture_shush and not gif_active:
        trigger_count_shush += 1
        gif_window_name = f"SHUSH GIF #{trigger_count_shush}"
        cv2.namedWindow(gif_window_name)
        cv2.moveWindow(
            gif_window_name,
            random.randint(RIGHT_START_X, RIGHT_START_X + POPUP_MAX_X_OFFSET),
            random.randint(POPUP_MIN_Y, POPUP_MAX_Y),
        )
        gif_active = True
        gif_index = 0
        gif_last_time = now

    if gif_active:
        t = time.time()
        if t - gif_last_time >= GIF_FRAME_INTERVAL:
            gif_last_time = t
            if gif_index < len(gif_frames):
                cv2.imshow(gif_window_name, gif_frames[gif_index])
                gif_index += 1
            else:
                gif_active = False  

    cv2.imshow("Gesture Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
