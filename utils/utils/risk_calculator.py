def calculate_risk(text):
    scam_keywords = [
        "payment", "fee", "registration",
        "whatsapp", "urgent", "earn money",
        "no experience", "work from home"
    ]

    found = [word for word in scam_keywords if word in text.lower()]

    risk_score = min(len(found) * 15, 100)

    return risk_score, found
