from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)

mp_pose = mp.solutions.pose


def calculate_body_asymmetry(landmarks):
    left_points = np.array([
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    ])

    right_points = np.array([
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    ])

    left_y_avg = np.mean(left_points)
    right_y_avg = np.mean(right_points)

    difference = abs(left_y_avg - right_y_avg)
    asymmetry_percentage = (difference / max(left_y_avg, right_y_avg)) * 100
    return asymmetry_percentage


@app.route('/body_asymmetry', methods=['POST'])
def calculate_asymmetry():
    if 'image' not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada."}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "O nome do arquivo está vazio."}), 400

    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    image = cv2.imread(file_path)
    if image is None:
        return jsonify({"error": "Imagem inválida ou não pôde ser processada."}), 400

    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return jsonify({"error": "Nenhum corpo detectado na imagem."}), 400

        landmarks = results.pose_landmarks.landmark
        asymmetry_percentage = calculate_body_asymmetry(landmarks)

        os.remove(file_path)

        if asymmetry_percentage > 20:
            return jsonify({
                "asymmetry_percentage": round(asymmetry_percentage, 2),
                "message": "Assimétrico"
            })
        else:
            return jsonify({
                "asymmetry_percentage": round(asymmetry_percentage, 2),
                "message": "Normal"
            })


if __name__ == '__main__':
    app.run(debug=True)
