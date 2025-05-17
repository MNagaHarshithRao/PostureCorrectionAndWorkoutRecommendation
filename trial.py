import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from collections import deque

# Load MoveNet Thunder
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
input_size = 256

# Buffer for smoothing keypoints
keypoint_history = deque(maxlen=5)

def detect_pose(frame):
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), input_size, input_size)
    input_img = tf.cast(img, dtype=tf.int32)
    outputs = movenet.signatures['serving_default'](input_img)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]
    return keypoints

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle

def average_keypoints(history):
    return np.mean(history, axis=0)

def analyze_posture(keypoints, h, w):
    get = lambda idx: [keypoints[idx][1] * w, keypoints[idx][0] * h]
    conf = lambda idx: keypoints[idx][2]

    # Use only upper body landmarks
    if conf(5) < 0.3 or conf(6) < 0.3 or conf(11) < 0.3 or conf(12) < 0.3:
        return [], "Insufficient keypoints for analysis."

    shoulder_l, shoulder_r = get(5), get(6)
    hip_l, hip_r = get(11), get(12)
    ear_l = get(3) if conf(3) > 0.3 else None
    ear_r = get(4) if conf(4) > 0.3 else None

    shoulder_mid = np.mean([shoulder_l, shoulder_r], axis=0)
    hip_mid = np.mean([hip_l, hip_r], axis=0)
    ear_mid = np.mean([p for p in [ear_l, ear_r] if p is not None], axis=0) if ear_l or ear_r else None

    # Measure posture
    spine_angle = calculate_angle(shoulder_mid, hip_mid, [hip_mid[0], hip_mid[1]+100])
    neck_angle = calculate_angle(shoulder_mid, ear_mid, [ear_mid[0], shoulder_mid[1]]) if ear_mid is not None else None
    shoulder_drop = abs(shoulder_l[1] - shoulder_r[1])
    lateral_shift = abs(shoulder_mid[0] - hip_mid[0])

    issues = []
    if spine_angle < 165:
        issues.append("Spine misalignment")
    if neck_angle and neck_angle > 35:
        issues.append("Forward head posture")
    if shoulder_drop > 20:
        issues.append("Shoulder imbalance")
    if lateral_shift > 25:
        issues.append("Lateral body lean")

    return issues, None

def suggest_exercise(issues):
    suggestions = {
        "Spine misalignment": "Try Cat-Cow or Wall Angels",
        "Forward head posture": "Do Chin Tucks against a wall",
        "Shoulder imbalance": "Strengthen upper traps and rear delts",
        "Lateral body lean": "Train obliques and trunk stabilizers"
    }
    if not issues:
        return "Posture looks good!"
    return "; ".join([suggestions[i] for i in issues if i in suggestions])

# OpenCV camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

font = cv2.FONT_HERSHEY_SIMPLEX
colors = {
    "Spine misalignment": (0, 0, 255),
    "Forward head posture": (255, 0, 0),
    "Shoulder imbalance": (128, 0, 128),
    "Lateral body lean": (0, 255, 255)
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints = detect_pose(input_frame)

    keypoint_history.append(keypoints)
    smoothed = average_keypoints(keypoint_history)

    for idx, kp in enumerate(smoothed):
        y, x, confidence = kp
        if confidence > 0.2:
            cx, cy = int(x * frame.shape[1]), int(y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    issues, warning = analyze_posture(smoothed, frame.shape[0], frame.shape[1])
    suggestion = suggest_exercise(issues)

    y_offset = 50
    if warning:
        cv2.putText(frame, warning, (30, y_offset), font, 0.7, (0, 0, 255), 2)
    elif issues:
        cv2.putText(frame, "Posture Issues:", (30, y_offset), font, 1, (0, 0, 255), 2)
        for issue in issues:
            y_offset += 35
            cv2.putText(frame, f"- {issue}", (50, y_offset), font, 0.8, colors.get(issue, (0, 255, 255)), 2)
    else:
        cv2.putText(frame, "Posture: Good!", (30, y_offset), font, 1.2, (0, 255, 0), 2)

    if not warning:
        cv2.putText(frame, f"Suggestion: {suggestion}", (30, y_offset + 60), font, 0.7, (200, 255, 200), 2)

    cv2.imshow("Upper Body Posture Monitor", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from collections import deque

# Load MoveNet Thunder
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
input_size = 256

# Buffer for smoothing keypoints
keypoint_history = deque(maxlen=5)

def detect_pose(frame):
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), input_size, input_size)
    input_img = tf.cast(img, dtype=tf.int32)
    outputs = movenet.signatures['serving_default'](input_img)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]
    return keypoints

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle

def average_keypoints(history):
    return np.mean(history, axis=0)

def analyze_posture(keypoints, h, w):
    get = lambda idx: [keypoints[idx][1] * w, keypoints[idx][0] * h]
    conf = lambda idx: keypoints[idx][2]

    # Use only upper body landmarks
    if conf(5) < 0.3 or conf(6) < 0.3 or conf(11) < 0.3 or conf(12) < 0.3:
        return [], "Insufficient keypoints for analysis."

    shoulder_l, shoulder_r = get(5), get(6)
    hip_l, hip_r = get(11), get(12)
    ear_l = get(3) if conf(3) > 0.3 else None
    ear_r = get(4) if conf(4) > 0.3 else None

    shoulder_mid = np.mean([shoulder_l, shoulder_r], axis=0)
    hip_mid = np.mean([hip_l, hip_r], axis=0)
    ear_mid = np.mean([p for p in [ear_l, ear_r] if p is not None], axis=0) if ear_l or ear_r else None

    # Measure posture
    spine_angle = calculate_angle(shoulder_mid, hip_mid, [hip_mid[0], hip_mid[1]+100])
    neck_angle = calculate_angle(shoulder_mid, ear_mid, [ear_mid[0], shoulder_mid[1]]) if ear_mid is not None else None
    shoulder_drop = abs(shoulder_l[1] - shoulder_r[1])
    lateral_shift = abs(shoulder_mid[0] - hip_mid[0])

    issues = []
    if spine_angle < 165:
        issues.append("Spine misalignment")
    if neck_angle and neck_angle > 35:
        issues.append("Forward head posture")
    if shoulder_drop > 20:
        issues.append("Shoulder imbalance")
    if lateral_shift > 25:
        issues.append("Lateral body lean")

    return issues, None

def suggest_exercise(issues):
    suggestions = {
        "Spine misalignment": "Try Cat-Cow or Wall Angels",
        "Forward head posture": "Do Chin Tucks against a wall",
        "Shoulder imbalance": "Strengthen upper traps and rear delts",
        "Lateral body lean": "Train obliques and trunk stabilizers"
    }
    if not issues:
        return "Posture looks good!"
    return "; ".join([suggestions[i] for i in issues if i in suggestions])

# OpenCV camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

font = cv2.FONT_HERSHEY_SIMPLEX
colors = {
    "Spine misalignment": (0, 0, 255),
    "Forward head posture": (255, 0, 0),
    "Shoulder imbalance": (128, 0, 128),
    "Lateral body lean": (0, 255, 255)
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints = detect_pose(input_frame)

    keypoint_history.append(keypoints)
    smoothed = average_keypoints(keypoint_history)

    for idx, kp in enumerate(smoothed):
        y, x, confidence = kp
        if confidence > 0.2:
            cx, cy = int(x * frame.shape[1]), int(y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    issues, warning = analyze_posture(smoothed, frame.shape[0], frame.shape[1])
    suggestion = suggest_exercise(issues)

    y_offset = 50
    if warning:
        cv2.putText(frame, warning, (30, y_offset), font, 0.7, (0, 0, 255), 2)
    elif issues:
        cv2.putText(frame, "Posture Issues:", (30, y_offset), font, 1, (0, 0, 255), 2)
        for issue in issues:
            y_offset += 35
            cv2.putText(frame, f"- {issue}", (50, y_offset), font, 0.8, colors.get(issue, (0, 255, 255)), 2)
    else:
        cv2.putText(frame, "Posture: Good!", (30, y_offset), font, 1.2, (0, 255, 0), 2)

    if not warning:
        cv2.putText(frame, f"Suggestion: {suggestion}", (30, y_offset + 60), font, 0.7, (200, 255, 200), 2)

    cv2.imshow("Upper Body Posture Monitor", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
