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

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'pose_landmarker_lite.task')

POSE_LANDMARKS = {
    'NOSE': 0,
    'LEFT_EAR': 7,
    'RIGHT_EAR': 8,
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,
    'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,
    'RIGHT_WRIST': 16,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,
    'RIGHT_ANKLE': 28,
}

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28),
    (0, 7), (0, 8),
]

SWING_PHASES = ['Address', 'Backswing', 'Top', 'Impact', 'Follow-Through', 'Finish']


def make_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def draw_pose_landmarks(image, landmarks, img_w, img_h, phase_label=None):
    points = {}
    for name, idx in POSE_LANDMARKS.items():
        lm = landmarks[idx]
        if lm.visibility > 0.3:
            x = int(lm.x * img_w)
            y = int(lm.y * img_h)
            points[idx] = (x, y)
            cv2.circle(image, (x, y), 5, (0, 220, 0), -1)
            cv2.circle(image, (x, y), 7, (255, 255, 255), 1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in points and end_idx in points:
            cv2.line(image, points[start_idx], points[end_idx], (0, 180, 255), 2)

    if phase_label:
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, 44), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
        cv2.putText(image, phase_label.upper(), (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 60, 220), 2, cv2.LINE_AA)
        cv2.putText(image, phase_label.upper(), (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)

    return points


def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def frame_to_b64(frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buffer).decode('utf-8')


def detect_phase_indices(wrist_y_series, total_frames):
    """
    Given normalized wrist Y over time (lower Y = hands higher),
    detect the 6 swing phase frame indices.
    """
    n = len(wrist_y_series)
    if n < 6:
        step = max(1, total_frames // 6)
        return [min(i * step, total_frames - 1) for i in range(6)]

    series = np.array(wrist_y_series)

    # Address: first 15% of frames
    address_idx = max(0, int(n * 0.05))

    # Top: minimum Y (hands highest) — look in first 65%
    search_end = int(n * 0.65)
    top_idx = int(np.argmin(series[:search_end])) if search_end > 0 else n // 3

    # Backswing midpoint
    backswing_idx = max(0, (address_idx + top_idx) // 2)

    # Impact: after top, find where wrists are lowest again (max Y)
    if top_idx < n - 1:
        impact_search = series[top_idx:]
        impact_offset = int(np.argmax(impact_search))
        impact_idx = top_idx + impact_offset
        # Clamp to not go too far
        impact_idx = min(impact_idx, int(n * 0.80))
    else:
        impact_idx = min(top_idx + 3, n - 1)

    # Follow-through midpoint
    follow_idx = min((impact_idx + n - 1) // 2, n - 2)

    # Finish: last 10% of frames
    finish_idx = max(follow_idx + 1, int(n * 0.90))
    finish_idx = min(finish_idx, n - 1)

    return [address_idx, backswing_idx, top_idx, impact_idx, follow_idx, finish_idx]


def analyze_phase_landmarks(landmarks, phase_name):
    """Generate phase-specific coaching feedback from landmarks."""
    feedback = []
    ls = landmarks[POSE_LANDMARKS['LEFT_SHOULDER']]
    rs = landmarks[POSE_LANDMARKS['RIGHT_SHOULDER']]
    lh = landmarks[POSE_LANDMARKS['LEFT_HIP']]
    rh = landmarks[POSE_LANDMARKS['RIGHT_HIP']]
    lw = landmarks[POSE_LANDMARKS['LEFT_WRIST']]
    rw = landmarks[POSE_LANDMARKS['RIGHT_WRIST']]
    lk = landmarks[POSE_LANDMARKS['LEFT_KNEE']]
    rk = landmarks[POSE_LANDMARKS['RIGHT_KNEE']]
    la = landmarks[POSE_LANDMARKS['LEFT_ANKLE']]
    ra = landmarks[POSE_LANDMARKS['RIGHT_ANKLE']]
    le = landmarks[POSE_LANDMARKS['LEFT_ELBOW']]
    re = landmarks[POSE_LANDMARKS['RIGHT_ELBOW']]

    wrist_avg_y = (lw.y + rw.y) / 2
    hip_avg_y = (lh.y + rh.y) / 2
    shoulder_avg_y = (ls.y + rs.y) / 2

    shoulder_tilt = abs(ls.y - rs.y)
    hip_tilt = abs(lh.y - rh.y)
    spine_offset = abs(((ls.x + rs.x) / 2) - ((lh.x + rh.x) / 2))

    if phase_name == 'Address':
        if shoulder_tilt < 0.04:
            feedback.append({"type": "good", "text": "Shoulders level at address — solid setup"})
        else:
            feedback.append({"type": "warning", "text": "Shoulders not level at address — square up before swinging"})

        if hip_tilt < 0.04:
            feedback.append({"type": "good", "text": "Hips aligned at setup"})
        else:
            feedback.append({"type": "warning", "text": "Hips misaligned — keep them parallel to target line"})

        if lk.visibility > 0.3 and rk.visibility > 0.3:
            lka = calc_angle([lh.x, lh.y], [lk.x, lk.y], [la.x, la.y])
            rka = calc_angle([rh.x, rh.y], [rk.x, rk.y], [ra.x, ra.y])
            avg_ka = (lka + rka) / 2
            if 150 <= avg_ka <= 175:
                feedback.append({"type": "good", "text": f"Good knee flex at address ({avg_ka:.0f}°)"})
            else:
                feedback.append({"type": "warning", "text": f"Knee angle {avg_ka:.0f}° — aim for 150–175° at address"})

        if spine_offset < 0.06:
            feedback.append({"type": "good", "text": "Spine upright and centered at setup"})
        else:
            feedback.append({"type": "warning", "text": "Spine tilting — stand taller at address"})

    elif phase_name == 'Backswing':
        if wrist_avg_y < hip_avg_y - 0.05:
            feedback.append({"type": "good", "text": "Hands are rising well in the backswing"})
        else:
            feedback.append({"type": "warning", "text": "Keep lifting the club — hands should be above hips here"})

        body_rotation = abs(ls.x - rs.x)
        if body_rotation > 0.12:
            feedback.append({"type": "good", "text": "Good shoulder turn in the backswing"})
        else:
            feedback.append({"type": "warning", "text": "Increase shoulder rotation on the backswing"})

        if hip_tilt < 0.08:
            feedback.append({"type": "good", "text": "Hips staying relatively level during backswing"})
        else:
            feedback.append({"type": "warning", "text": "Hips tilting too much — resist with your lower body"})

    elif phase_name == 'Top':
        if wrist_avg_y < shoulder_avg_y:
            feedback.append({"type": "good", "text": "Club is above shoulder height at the top — full swing achieved"})
        elif wrist_avg_y < hip_avg_y:
            feedback.append({"type": "warning", "text": "Club not reaching full height — work on flexibility and turn"})
        else:
            feedback.append({"type": "warning", "text": "Backswing incomplete — try a fuller shoulder rotation"})

        if re.visibility > 0.3:
            elbow_angle = calc_angle([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
            if 70 <= elbow_angle <= 110:
                feedback.append({"type": "good", "text": f"Trail elbow well positioned at the top ({elbow_angle:.0f}°)"})
            else:
                feedback.append({"type": "warning", "text": f"Trail elbow angle {elbow_angle:.0f}° — aim for roughly 90° at top"})

        body_rotation = abs(ls.x - rs.x)
        if body_rotation > 0.15:
            feedback.append({"type": "good", "text": "Excellent shoulder coil at the top of the swing"})
        else:
            feedback.append({"type": "warning", "text": "Increase shoulder turn — aim for 90° rotation at the top"})

    elif phase_name == 'Impact':
        if wrist_avg_y >= hip_avg_y - 0.05:
            feedback.append({"type": "good", "text": "Hands returning to impact zone — solid ball-striking position"})
        else:
            feedback.append({"type": "warning", "text": "Hands too high at impact — drive them down toward the ball"})

        if shoulder_tilt < 0.08:
            feedback.append({"type": "good", "text": "Shoulders rotating through at impact"})
        else:
            feedback.append({"type": "warning", "text": "Keep rotating — shoulders should be squaring up at impact"})

        hip_lead = (lh.x + rh.x) / 2 - (ls.x + rs.x) / 2
        if abs(hip_lead) > 0.02:
            feedback.append({"type": "good", "text": "Hips leading the downswing — excellent sequencing"})
        else:
            feedback.append({"type": "warning", "text": "Initiate downswing with hips clearing toward the target"})

    elif phase_name == 'Follow-Through':
        if wrist_avg_y < hip_avg_y:
            feedback.append({"type": "good", "text": "Arms extending well into the follow-through"})
        else:
            feedback.append({"type": "warning", "text": "Keep the arms extending through the ball after impact"})

        if lk.visibility > 0.3 and rk.visibility > 0.3:
            lka = calc_angle([lh.x, lh.y], [lk.x, lk.y], [la.x, la.y])
            if lka > 160:
                feedback.append({"type": "good", "text": "Lead leg straightening nicely into the follow-through"})
            else:
                feedback.append({"type": "warning", "text": "Drive into a firmer lead leg as you follow through"})

        body_rotation = abs(ls.x - rs.x)
        if body_rotation > 0.1:
            feedback.append({"type": "good", "text": "Good body rotation through the shot"})
        else:
            feedback.append({"type": "warning", "text": "Keep rotating — don't stop the body at the ball"})

    elif phase_name == 'Finish':
        if wrist_avg_y < shoulder_avg_y:
            feedback.append({"type": "good", "text": "High finish position — club fully through the swing"})
        else:
            feedback.append({"type": "warning", "text": "Finish higher — swing the club all the way around"})

        if spine_offset < 0.12:
            feedback.append({"type": "good", "text": "Balanced finish — weight transferred well"})
        else:
            feedback.append({"type": "warning", "text": "Finish in balance — hold the pose to check your weight transfer"})

        if shoulder_tilt > 0.06:
            feedback.append({"type": "good", "text": "Shoulders have fully rotated through to the finish"})
        else:
            feedback.append({"type": "warning", "text": "Commit to a full shoulder turn all the way to the finish"})

    return feedback


def process_swing_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Sample up to 80 frames spread across the video
    sample_count = min(80, total_frames)
    sample_step = max(1, total_frames // sample_count)

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_step == 0:
            frames.append((frame_idx, frame.copy()))
        frame_idx += 1
    cap.release()

    if not frames:
        return {"error": "No frames extracted from video"}

    # Run pose detection on all sampled frames
    wrist_y_series = []
    landmarks_per_frame = []

    with make_landmarker() as landmarker:
        for _, frame in frames:
            img_h, img_w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = landmarker.detect(mp_img)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lms = result.pose_landmarks[0]
                lw = lms[POSE_LANDMARKS['LEFT_WRIST']]
                rw = lms[POSE_LANDMARKS['RIGHT_WRIST']]
                wrist_y = (lw.y + rw.y) / 2
                wrist_y_series.append(wrist_y)
                landmarks_per_frame.append(lms)
            else:
                # No detection — interpolate with last known or default
                wrist_y_series.append(wrist_y_series[-1] if wrist_y_series else 0.6)
                landmarks_per_frame.append(None)

    if all(lm is None for lm in landmarks_per_frame):
        return {"error": "No pose detected in any frame. Ensure your full body is visible."}

    # Detect phase frame indices
    phase_indices = detect_phase_indices(wrist_y_series, len(frames))

    # Build phase results
    phase_results = []
    for i, (phase_name, phase_frame_idx) in enumerate(zip(SWING_PHASES, phase_indices)):
        orig_idx, frame = frames[phase_frame_idx]
        lms = landmarks_per_frame[phase_frame_idx]

        annotated = frame.copy()
        img_h, img_w = annotated.shape[:2]

        if lms:
            draw_pose_landmarks(annotated, lms, img_w, img_h, phase_name)
            phase_feedback = analyze_phase_landmarks(lms, phase_name)
        else:
            # Try adjacent frames for landmarks
            phase_feedback = [{"type": "warning", "text": "Pose not clearly detected in this phase"}]
            cv2.putText(annotated, phase_name.upper(), (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        phase_results.append({
            "phase": phase_name,
            "image": f"data:image/jpeg;base64,{frame_to_b64(annotated)}",
            "feedback": phase_feedback
        })

    # Overall swing summary
    overall_feedback = build_overall_feedback(wrist_y_series, landmarks_per_frame)

    return {
        "phases": phase_results,
        "overall": overall_feedback,
        "frame_count": total_frames,
        "duration_sec": round(total_frames / fps, 1)
    }


def build_overall_feedback(wrist_y_series, landmarks_list):
    feedback = []

    # Swing tempo: smoothness of wrist trajectory
    valid_y = [y for y in wrist_y_series if y is not None]
    if len(valid_y) > 4:
        diffs = np.diff(valid_y)
        smoothness = np.std(diffs)
        if smoothness < 0.025:
            feedback.append({"type": "good", "text": "Smooth, consistent swing tempo throughout"})
        else:
            feedback.append({"type": "warning", "text": "Work on swing tempo — try a more even, rhythmic pace"})

        # Swing range: how high do hands get vs start
        start_y = np.mean(valid_y[:3])
        min_y = np.min(valid_y)
        swing_range = start_y - min_y
        if swing_range > 0.25:
            feedback.append({"type": "good", "text": "Full range of motion — excellent club path height"})
        elif swing_range > 0.10:
            feedback.append({"type": "warning", "text": "Swing range could be fuller — aim for a more complete backswing"})
        else:
            feedback.append({"type": "warning", "text": "Swing appears limited — ensure you're making a full backswing"})

    # Check spine consistency across frames
    valid_lms = [lm for lm in landmarks_list if lm is not None]
    if len(valid_lms) > 4:
        spine_offsets = []
        for lm in valid_lms:
            ls = lm[POSE_LANDMARKS['LEFT_SHOULDER']]
            rs = lm[POSE_LANDMARKS['RIGHT_SHOULDER']]
            lh = lm[POSE_LANDMARKS['LEFT_HIP']]
            rh = lm[POSE_LANDMARKS['RIGHT_HIP']]
            offset = abs(((ls.x + rs.x) / 2) - ((lh.x + rh.x) / 2))
            spine_offsets.append(offset)

        avg_spine = np.mean(spine_offsets)
        if avg_spine < 0.08:
            feedback.append({"type": "good", "text": "Spine angle maintained well throughout the swing"})
        else:
            feedback.append({"type": "warning", "text": "Spine angle inconsistent — focus on keeping your posture throughout"})

    return feedback


def analyze_golf_swing(image_data):
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, "Failed to decode image"

    img_h, img_w, _ = img_bgr.shape
    img_result = img_bgr.copy()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    feedback = []

    with make_landmarker() as landmarker:
        detection_result = landmarker.detect(mp_image)

        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            feedback.append({"type": "warning", "text": "No pose detected - ensure full body is visible in frame"})
            _, buffer = cv2.imencode('.jpg', img_result)
            return base64.b64encode(buffer).decode('utf-8'), feedback

        landmarks = detection_result.pose_landmarks[0]
        draw_pose_landmarks(img_result, landmarks, img_w, img_h)
        feedback = analyze_phase_landmarks(landmarks, 'Address')

    _, buffer = cv2.imencode('.jpg', img_result)
    return base64.b64encode(buffer).decode('utf-8'), feedback


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    image_data = data['image']
    if ',' in image_data:
        image_data = image_data.split(',')[1]

    try:
        result_image, feedback = analyze_golf_swing(image_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if result_image is None:
        return jsonify({"error": str(feedback)}), 500

    return jsonify({
        "image": f"data:image/jpeg;base64,{result_image}",
        "feedback": feedback
    })


@app.route('/analyze-video', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video provided"}), 400

    video_file = request.files['video']

    suffix = '.webm'
    if video_file.filename and '.' in video_file.filename:
        suffix = '.' + video_file.filename.rsplit('.', 1)[-1]

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            video_file.save(tmp.name)
            tmp_path = tmp.name

        result = process_swing_video(tmp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
