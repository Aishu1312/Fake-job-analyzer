# 🕵️ Fake Job Analyzer

An AI-powered web application that detects fraudulent job postings using **Machine Learning** and **Natural Language Processing**.

Built with **Streamlit**, **Scikit-learn**, and **TF-IDF Vectorization**.

**Author:** Aishwarya Lala  
**College:** St. Vincent Pallotti College of Engineering and Technology

---

## 🚀 Live Demo

Deploy instantly on [Streamlit Cloud](https://share.streamlit.io) — see deployment steps below.

---

## 📁 Project Structure

```
fake-job-analyzer/
│
├── streamlit_app.py           ← Main Streamlit app (entry point)
├── requirements.txt           ← Python dependencies
│
├── .streamlit/
│   └── config.toml            ← Streamlit server configuration
│
├── utils/
│   ├── feature_extractor.py   ← Text cleaning and preprocessing
│   └── risk_calculator.py     ← Rule-based scam keyword detector
│
├── model/
│   └── train_model.py         ← Model training script (Logistic Regression + TF-IDF)
│
└── data/
    └── create_dataset.py      ← Synthetic training dataset generator
```

---

## ⚙️ How It Works

1. **Text Preprocessing** — The job description is lowercased and cleaned using regex.
2. **TF-IDF Vectorization** — Text is converted into numerical feature vectors.
3. **ML Classification** — A trained Logistic Regression model predicts Real or Fake.
4. **Risk Scoring** — A rule-based engine scans for 25 known scam keywords and calculates a risk score.
5. **Results** — The app displays the verdict, confidence probability, risk score, and suspicious indicators found.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI Framework | Streamlit |
| ML Model | Logistic Regression (Scikit-learn) |
| Feature Extraction | TF-IDF Vectorizer |
| Data Processing | Pandas, NumPy |
| Language | Python 3.11 |

---

## 📦 Installation (Local)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/fake-job-analyzer.git
cd fake-job-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run streamlit_app.py
```

The model trains automatically on first launch using the built-in dataset.

---

## 🌐 Deploy on Streamlit Cloud (Free)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** → Connect your GitHub repo.
4. Set **Main file path** to `streamlit_app.py`.
5. Click **Deploy**.

No extra configuration needed — the model auto-trains on first run.

---

## 🔍 Features

- ✅ Real-time fake job detection
- 📊 ML confidence score (%)
- ⚠️ Risk score with color-coded bar
- 🔎 Suspicious keyword breakdown
- 📋 One-click sample job loaders (real & fake)
- 📖 Sidebar with how-it-works guide

---

## ⚠️ Disclaimer

This tool assists decision-making and should not be used as the sole verification method. Always research the company independently.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
