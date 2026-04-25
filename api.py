from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text"]
    vec = vectorizer.transform([data])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][1]

    return jsonify({
        "prediction": int(pred),
        "risk_score": float(prob)
    })

if __name__ == "__main__":
    app.run()
