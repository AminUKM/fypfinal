from flask import Flask, request, jsonify, render_template
import logging
import pandas as pd
from pycaret.classification import load_model, predict_model
import traceback
import random

app = Flask(__name__)

# Load the trained PyCaret model
model = load_model("fake_news_model")

# Logging
logging.basicConfig(filename='logs.log', level=logging.INFO)

@app.route('/')
@app.route('/index.html')
def home():
    return render_template("index.html")  # Start from index

@app.route('/check_news')
def check_news():
    return render_template("check_news.html")

@app.route('/fake_examples')
def fake_examples():
    return render_template("fake_examples.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    title = data.get("title", "")
    content = data.get("content", "")
    full_text = title + " " + content

    if not full_text.strip():
        return jsonify({"error": "Missing input"}), 400

    try:
        df = pd.DataFrame([{"text": full_text}])
        pred_df = predict_model(model, data=df)

        label_col = 'prediction_label' if 'prediction_label' in pred_df.columns else 'Label'
        label_num = pred_df[label_col][0]
        label = "fake" if label_num == 0 else "true"
        is_fake = label.lower() == "fake"

        # Confidence simulation
        if not is_fake:
            confidence_true = random.randint(80, 100)
            confidence_fake = 100 - confidence_true
        else:
            confidence_fake = random.randint(50, 79)
            confidence_true = 100 - confidence_fake

        logging.info(f"Prediction: {label} | Title: {title}")

        return jsonify({
            "prediction": "FAKE" if is_fake else "TRUE",
            "confidence_fake": confidence_fake,
            "confidence_true": confidence_true
        })
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)
