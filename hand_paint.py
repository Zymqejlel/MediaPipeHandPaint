import cv2
import mediapipe as mp
import numpy as np
import os
# 减少警告刷屏
os.environ['GLOG_minloglevel'] = '2'

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

canvas = np.zeros((480, 640, 3), np.uint8)
last_point = None

# 判断手指抬起（中指/无名指/小指）
def is_finger_up(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y

# 计算两点距离，用来判断拇指和食指贴在一起（OK手势）
def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

win_name = "MediaPipe Hand Paint"
cv2.namedWindow(win_name)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ 0号摄像头打不开,尝试1号")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ 摄像头全部打不开，请检查权限/占用")
        exit()
print("✅ 摄像头启动成功")

exit_flag = False
exit_stable_count = 0
need_stable_frame = 5  # 连续5帧识别到OK，才退出

while cap.isOpened() and not exit_flag:
    try:
        ret, frame = cap.read()
        retry = 0
        while not ret and retry < 3:
            ret, frame = cap.read()
            retry += 1
        if not ret:
            print("❌ 多次读取画面失败，退出")
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_img)

        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                lms = hand_lms.landmark

                index_up = is_finger_up(lms, 8, 6)
                middle_up = is_finger_up(lms, 12, 10)
                ring_up = is_finger_up(lms, 16, 14)
                pinky_up = is_finger_up(lms, 20, 18)

                # 拇指、食指指尖距离
                thumb_index_dist = get_distance(lms[4], lms[8])

                # 手势1：只伸食指，其余三指弯曲 → 画线
                if index_up and not middle_up and not ring_up and not pinky_up:
                    x = int(lms[8].x * w)
                    y = int(lms[8].y * h)
                    current_point = (x, y)
                    if last_point is not None:
                        cv2.line(canvas, last_point, current_point, (255, 0, 255), 4)
                    last_point = current_point
                    exit_stable_count = 0  # 重置防抖计数
                # 手势2：全部手指弯曲，握拳 → 清空画布
                elif not index_up and not middle_up and not ring_up and not pinky_up:
                    canvas = np.zeros((480, 640, 3), np.uint8)
                    last_point = None
                    exit_stable_count = 0
                # 手势3：OK手势：拇指食指靠很近 + 中指、无名指、小指伸直张开
                elif thumb_index_dist < 0.08 and middle_up and ring_up and pinky_up:
                    exit_stable_count += 1
                    if exit_stable_count >= need_stable_frame:
                        exit_flag = True
                else:
                    last_point = None
                    exit_stable_count = 0

        output = cv2.addWeighted(frame, 1, canvas, 0.6, 0)
        cv2.imshow(win_name, output)

        # 右上角叉号关闭
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        # ESC兜底退出
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    except Exception as e:
        print("帧异常跳过：", e)
        continue

# 统一收尾释放资源，优雅退出，不会窗口残影卡死
cap.release()
cv2.destroyAllWindows()