import flask
from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import requests
from urllib.parse import urlparse
from io import BytesIO

app = Flask(__name__)
mp_pose = mp.solutions.pose

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        raise ValueError(f"Error downloading image: {str(e)}")

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
    if 'image_url' not in request.json:
        return jsonify({"error": "Nenhuma URL de imagem fornecida."}), 400
    
    image_url = request.json['image_url']
    
    if not is_valid_url(image_url):
        return jsonify({"error": "URL de imagem inválida."}), 400
    
    try:
        image = download_image(image_url)
        
        if image is None:
            return jsonify({"error": "Imagem inválida ou não pôde ser processada."}), 400
        
        with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return jsonify({"error": "Nenhum corpo detectado na imagem."}), 400
        
        landmarks = results.pose_landmarks.landmark
        asymmetry_percentage = calculate_body_asymmetry(landmarks)
        
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
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)