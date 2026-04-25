from flask import Flask, render_template, request
import joblib
from utils.feature_extractor import extract_features
from utils.risk_calculator import calculate_risk, risk_level

app = Flask(__name__)

# Load model
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        text = request.form['description']
        email = request.form['email']

        # ML prediction
        X = vectorizer.transform([text])
        ml_prob = model.predict_proba(X)[0][1]

        # Feature extraction
        features = extract_features(text, email)

        # Risk calculation
        score = calculate_risk(features, ml_prob)
        level = risk_level(score)

        result = {
            "score": round(score, 2),
            "level": level,
            "features": features
        }

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
