import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('best_model_xgb.pkl', 'rb'))

@app.route('/')
def home():
    return "Welcome to the Concrete Strength Prediction API Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        #  JSON data from request
        data = request.json

        # Extract features 
        cement = float(data['Cement'])
        slag = float(data['Slag'])
        fly_ash = float(data['fly_ash'])
        water = float(data['Water'])
        superplasticizer = float(data['superplasticizer'])
        coarse_aggregate = float(data['coarse_aggregate'])
        fine_aggregate = float(data['fine_aggregate'])
        age = int(data['age'])
        features = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]])
        prediction = model.predict(features)[0]

        return jsonify({
            "strength": prediction,
            "statusCode": 200
        }), 200

    except KeyError as e:
        return jsonify({"message": f"Missing or invalid data: {str(e)}", "statusCode": 400}), 400

    except Exception as e:
        return jsonify({"message": str(e), "statusCode": 500}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
