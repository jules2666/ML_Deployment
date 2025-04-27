
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle
with open('app/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if 'features' not in data:
        return jsonify({'error': 'Missing "features" key'}), 400

    features = np.array(data['features'])

    if features.ndim == 1:
        features = features.reshape(1, -1)

    if not all(len(f) == 4 for f in features):
        return jsonify({'error': 'Each input must have 4 float values'}), 400

    prediction = model.predict(features).tolist()
    confidence = model.predict_proba(features).max(axis=1).tolist()

    if len(prediction) == 1:
        return jsonify({
            'prediction': prediction[0],
            'confidence': confidence[0]
        })
    else:
        return jsonify({
            'predictions': prediction,
            'confidences': confidence
        })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})
