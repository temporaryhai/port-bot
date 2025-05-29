import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import nltk

# Setup logging
logging.basicConfig(level=logging.INFO)

# Download NLTK data quietly if not already downloaded
nltk.download('punkt', quiet=True)

# Load model and vectorizer paths from environment variables for flexibility
MODEL_PATH = os.getenv('MODEL_PATH', 'model.pkl')
VECTORIZER_PATH = os.getenv('VECTORIZER_PATH', 'vectorizer.pkl')

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    logging.error("Model or vectorizer file not found. Please ensure they are present.")
    raise FileNotFoundError("Model or vectorizer file not found. Please ensure they are present.")

# Load model and vectorizer
classifier = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def preprocess(text: str) -> str:
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join(tokens)

app = Flask(__name__)
CORS(app)  # You should restrict CORS origins in production

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(force=True)
        user_input = data.get('message', '').strip()

        if not user_input:
            return jsonify({"response": "Please type something!"}), 400

        processed_input = preprocess(user_input)
        input_vector = vectorizer.transform([processed_input])
        prediction = classifier.predict(input_vector)
        response = prediction[0]

        return jsonify({"response": response})

    except Exception:
        logging.exception("Error during request processing:")
        return jsonify({"response": "An internal error occurred. Please try again later."}), 500

if __name__ == "__main__":
    # In production, use gunicorn or another WSGI server instead of Flask's built-in server
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5505)), debug=False)
