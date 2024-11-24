from flask import Flask, request, jsonify, send_file
import cv2
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Utilitário para desenhar landmarks


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


@app.route('/test_landmarks', methods=['POST'])
def test_landmarks():
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

        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )

        annotated_path = os.path.join("uploads", "annotated_" + file.filename)
        cv2.imwrite(annotated_path, annotated_image)

        landmarks = results.pose_landmarks.landmark
        asymmetry_percentage = calculate_body_asymmetry(landmarks)

        os.remove(file_path)

        return jsonify({
            "asymmetry_percentage": round(asymmetry_percentage, 2),
            "annotated_image_url": f"/view_image/{'annotated_' + file.filename}"
        })


@app.route('/view_image/<filename>', methods=['GET'])
def view_image(filename):
    file_path = os.path.join("uploads", filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Imagem não encontrada."}), 404
    return send_file(file_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)
