SCAM_KEYWORDS = [
    "payment",
    "fee",
    "registration",
    "whatsapp",
    "urgent",
    "earn money",
    "no experience",
    "work from home",
    "guaranteed income",
    "wire transfer",
    "western union",
    "money order",
    "upfront",
    "investment required",
    "send money",
    "training fee",
    "background check fee",
    "unlimited earnings",
    "passive income",
    "be your own boss",
    "free laptop",
    "free iphone",
    "data entry",
    "copy paste",
    "part time from home",
]

KEYWORD_DESCRIPTIONS = {
    "payment":            "Requests payment from applicant",
    "fee":                "Mentions fees or charges",
    "registration":       "Requires registration fee",
    "whatsapp":           "Uses WhatsApp for contact (unusual for legit jobs)",
    "urgent":             "Creates false urgency",
    "earn money":         "Vague earning promises",
    "no experience":      "Claims no experience needed (suspicious if paired with high pay)",
    "work from home":     "Unverified work-from-home offer",
    "guaranteed income":  "Promises guaranteed income",
    "wire transfer":      "Mentions wire transfer",
    "western union":      "Mentions Western Union payment",
    "money order":        "Requests money order",
    "upfront":            "Requires upfront payment",
    "investment required":"Requires financial investment",
    "send money":         "Asks to send money",
    "training fee":       "Charges training fee",
    "background check fee":"Charges for background check",
    "unlimited earnings": "Claims unlimited earnings",
    "passive income":     "Promises passive income",
    "be your own boss":   "Uses 'be your own boss' pitch",
    "free laptop":        "Promises free laptop",
    "free iphone":        "Promises free iPhone",
    "data entry":         "Simple data entry tasks (often low-quality jobs)",
    "copy paste":         "Copy-paste tasks (common in scams)",
    "part time from home":"Unverified part-time work-from-home offer",
}


def calculate_risk(text: str):
    text_lower = text.lower()
    found_keywords = [kw for kw in SCAM_KEYWORDS if kw in text_lower]
    risk_score = min(len(found_keywords) * 12, 100)
    reasons = [KEYWORD_DESCRIPTIONS[kw] for kw in found_keywords]
    return risk_score, reasons
