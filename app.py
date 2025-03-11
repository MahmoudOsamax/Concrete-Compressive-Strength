import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

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
        slag = float(data['Blast Furnace Slag'])
        fly_ash = float(data['Fly Ash'])
        water = float(data['Water'])
        superplasticizer = float(data['Superplasticizer'])
        coarse_aggregate = float(data['Coarse Aggregate'])
        fine_aggregate = float(data['Fine Aggregate'])
        age = int(data['Age (day)'])

    
        features = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]])

        prediction = float(model.predict(features)[0])

        return jsonify({
            "Concrete_Compressive_Strength": prediction,
            "statusCode": 200
        }), 200

    except KeyError as e:
        return jsonify({"message": f"Missing or invalid data: {str(e)}", "statusCode": 400}), 400

    except Exception as e:
        return jsonify({"message": str(e), "statusCode": 500}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
