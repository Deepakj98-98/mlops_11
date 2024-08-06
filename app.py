from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trapythined model
model = joblib.load("linear_regression_best.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input JSON
    data = request.get_json(force=True)
    median_income = data.get('median_income', None)

    if median_income is None:
        return jsonify({'error': 'median_income is required'}), 400

    # Reshape and predict
    median_income_array = np.array(median_income).reshape(-1, 1)
    predictions = model.predict(median_income_array)
    
    # Return the predictions as JSON
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)