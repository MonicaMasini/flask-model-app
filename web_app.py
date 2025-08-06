from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the model
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Model is ready to predict!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({'error': 'Missing features in request'}), 400

    try:
        prediction = model.predict([data['features']])
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
