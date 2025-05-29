from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import nltk
import os

# Download NLTK tokenizer if needed
nltk.download('punkt')

# Load model and vectorizer
model_path = 'model.pkl'
vectorizer_path = 'vectorizer.pkl'

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Trained model or vectorizer not found. Please run train_model.py first.")

classifier = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Preprocess function
def pre(text):
    words = nltk.word_tokenize(text.lower())
    return ' '.join(words)

# Set up Flask app
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data.get('message', '').strip()

        if not user_input:
            return jsonify({"response": "Please type something!"}), 400

        processed_input = pre(user_input)
        input_vector = vectorizer.transform([processed_input])
        response = classifier.predict(input_vector)[0]

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5505)
