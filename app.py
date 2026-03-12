import cv2
import mediapipe as mp
import numpy as np
import base64
import os
import tempfile
from flask import Flask, render_template, request, jsonify
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_lite.task")

POSE_LANDMARKS = {
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
}

SWING_PHASES = ["Address", "Backswing", "Top", "Impact", "Follow-Through", "Finish"]


def make_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)

    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    return mp_vision.PoseLandmarker.create_from_options(options)


def draw_pose_landmarks(image, landmarks, img_w, img_h, label=None):
    for idx in POSE_LANDMARKS.values():
        lm = landmarks[idx]
        if lm.visibility > 0.3:
            x = int(lm.x * img_w)
            y = int(lm.y * img_h)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    if label:
        cv2.putText(
            image,
            label.upper(),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )


def frame_to_b64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def detect_phase_indices(series, total_frames):
    n = len(series)

    if n < 6:
        step = max(1, total_frames // 6)
        return [min(i * step, total_frames - 1) for i in range(6)]

    series = np.array(series)

    address = int(n * 0.05)
    top = int(np.argmin(series[: int(n * 0.65)]))
    backswing = (address + top) // 2

    impact_offset = int(np.argmax(series[top:]))
    impact = min(top + impact_offset, int(n * 0.8))

    follow = min((impact + n - 1) // 2, n - 2)
    finish = min(int(n * 0.9), n - 1)

    return [address, backswing, top, impact, follow, finish]


def analyze_phase_landmarks(landmarks, phase):
    feedback = []

    lw = landmarks[POSE_LANDMARKS["LEFT_WRIST"]]
    rw = landmarks[POSE_LANDMARKS["RIGHT_WRIST"]]

    wrist_height = (lw.y + rw.y) / 2

    if phase == "Address":
        feedback.append({"type": "good", "text": "Setup detected"})

    if phase == "Top":
        if wrist_height < 0.4:
            feedback.append({"type": "good", "text": "Full backswing height"})
        else:
            feedback.append({"type": "warning", "text": "Raise club higher at top"})

    if phase == "Impact":
        feedback.append({"type": "good", "text": "Impact frame detected"})

    return feedback


def build_overall_feedback(wrist_series, landmarks_list):
    feedback = []

    if len(wrist_series) > 5:
        diffs = np.diff(wrist_series)

        if np.std(diffs) < 0.03:
            feedback.append({"type": "good", "text": "Smooth swing tempo"})
        else:
            feedback.append({"type": "warning", "text": "Swing tempo inconsistent"})

    return feedback


def process_swing_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Could not open video"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    sample_count = min(80, total_frames)
    sample_step = max(1, total_frames // sample_count)

    frames = []
    wrist_series = []
    landmarks_list = []

    frame_idx = 0

    with make_landmarker() as landmarker:

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_idx % sample_step == 0:

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                result = landmarker.detect(mp_img)

                if result.pose_landmarks:

                    lms = result.pose_landmarks[0]

                    lw = lms[POSE_LANDMARKS["LEFT_WRIST"]]
                    rw = lms[POSE_LANDMARKS["RIGHT_WRIST"]]

                    wrist_series.append((lw.y + rw.y) / 2)

                    landmarks_list.append(lms)

                    frames.append((frame_idx, frame.copy()))

            frame_idx += 1

    cap.release()

    if not frames:
        return {"error": "No frames processed"}

    phases = detect_phase_indices(wrist_series, len(frames))

    phase_results = []

    for phase_name, idx in zip(SWING_PHASES, phases):

        frame = frames[idx][1].copy()
        img_h, img_w = frame.shape[:2]

        lms = landmarks_list[idx]

        draw_pose_landmarks(frame, lms, img_w, img_h, phase_name)

        feedback = analyze_phase_landmarks(lms, phase_name)

        phase_results.append(
            {
                "phase": phase_name,
                "image": f"data:image/jpeg;base64,{frame_to_b64(frame)}",
                "feedback": feedback,
            }
        )

    overall = build_overall_feedback(wrist_series, landmarks_list)

    return {
        "phases": phase_results,
        "overall": overall,
        "frame_count": total_frames,
        "duration_sec": round(total_frames / fps, 1),
    }


def analyze_golf_swing(image_data):

    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None, "Image decode failed"

    h, w = img.shape[:2]

    with make_landmarker() as landmarker:

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_img)

        if not result.pose_landmarks:
            return None, "No pose detected"

        lms = result.pose_landmarks[0]

        draw_pose_landmarks(img, lms, w, h)

        feedback = analyze_phase_landmarks(lms, "Address")

    return frame_to_b64(img), feedback


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()

    image_data = data["image"]

    if "," in image_data:
        image_data = image_data.split(",")[1]

    image, feedback = analyze_golf_swing(image_data)

    return jsonify(
        {"image": f"data:image/jpeg;base64,{image}", "feedback": feedback}
    )


@app.route("/analyze-video", methods=["POST"])
def analyze_video():

    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"})

    video = request.files["video"]

    suffix = ".webm"

    if "." in video.filename:
        suffix = "." + video.filename.rsplit(".", 1)[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:

        video.save(tmp.name)

        path = tmp.name

    result = process_swing_video(path)

    os.remove(path)

    return jsonify(result)


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
