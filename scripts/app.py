#!/usr/bin/python3

from flask import Flask, request, jsonify
from src.predictor import Predictor

app = Flask(__name__)
predictor = Predictor('best_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = predictor.predict(data)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
