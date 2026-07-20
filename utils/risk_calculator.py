import re

SCAM_KEYWORDS = [
    "payment", "fee", "registration", "whatsapp", "urgent",
    "earn money", "no experience", "work from home", "guaranteed income",
    "wire transfer", "western union", "money order", "upfront",
    "investment required", "send money", "training fee",
    "background check fee", "unlimited earnings", "passive income",
    "be your own boss", "free laptop", "free iphone",
    "data entry", "copy paste", "part time from home",
]

KEYWORD_DESCRIPTIONS = {
    "payment": "Requests payment from applicant",
    "fee": "Mentions fees or charges",
    "registration": "Requires registration fee",
    "whatsapp": "Uses WhatsApp for contact (unusual for legit jobs)",
    "urgent": "Creates false urgency",
    "earn money": "Vague earning promises",
    "no experience": "Claims no experience needed (suspicious if paired with high pay)",
    "work from home": "Unverified work-from-home offer",
    "guaranteed income": "Promises guaranteed income",
    "wire transfer": "Mentions wire transfer",
    "western union": "Mentions Western Union payment",
    "money order": "Requests money order",
    "upfront": "Requires upfront payment",
    "investment required": "Requires financial investment",
    "send money": "Asks to send money",
    "training fee": "Charges training fee",
    "background check fee": "Charges for background check",
    "unlimited earnings": "Claims unlimited earnings",
    "passive income": "Promises passive income",
    "be your own boss": "Uses 'be your own boss' pitch",
    "free laptop": "Promises free laptop",
    "free iphone": "Promises free iPhone",
    "data entry": "Simple data entry tasks (often low-quality jobs)",
    "copy paste": "Copy-paste tasks (common in scams)",
    "part time from home": "Unverified part-time work-from-home offer",
}

URGENCY_KW = ["urgent", "immediate start", "last", "act now", "limited time", "hiring now"]
SALARY_KW = ["unlimited", "passive income", "guaranteed", "no experience", "earn money"]
CONTACT_KW = ["whatsapp", "wire transfer", "western union", "telegram"]

def calculate_granular_risk(text: str):
    text_lower = text.lower()
    
    # Base Keyword Risk
    found_keywords = [kw for kw in SCAM_KEYWORDS if kw in text_lower]
    reasons = [KEYWORD_DESCRIPTIONS[kw] for kw in found_keywords]
    base_risk = min(len(found_keywords) * 12, 100)
    
    # Urgency Manipulation
    urgency_flags = sum(1 for kw in URGENCY_KW if kw in text_lower)
    urgency_score = min(urgency_flags * 25, 100)
    if urgency_score > 0 and "High urgency language detected" not in reasons:
        reasons.append("High urgency language detected")
        
    # Salary Manipulation
    salary_flags = sum(1 for kw in SALARY_KW if kw in text_lower)
    salary_score = min(salary_flags * 20, 100)
    if salary_score > 0 and "Unrealistic salary or earnings promises" not in reasons:
        reasons.append("Unrealistic salary or earnings promises")
        
    # Contact Authenticity (100 is good, 0 is bad)
    contact_flags = sum(1 for kw in CONTACT_KW if kw in text_lower)
    contact_authenticity = max(100 - (contact_flags * 50), 0)
    if contact_authenticity < 100 and "Suspicious contact methods (e.g. WhatsApp/Telegram)" not in reasons:
        reasons.append("Suspicious contact methods (e.g. WhatsApp/Telegram)")

    # Grammar Quality (Heuristics: ALL CAPS, excessive punctuation !!!)
    caps_count = len(re.findall(r'[A-Z]{3,}', text))
    exclamations = len(re.findall(r'!{2,}', text))
    grammar_penalty = min((caps_count * 5) + (exclamations * 10), 100)
    grammar_quality = 100 - grammar_penalty
    if grammar_quality < 50 and "Poor grammar / excessive capitalization" not in reasons:
        reasons.append("Poor grammar / excessive capitalization or punctuation")
        
    # Missing Information
    words = text.split()
    missing_info = 100 if len(words) < 30 else (50 if len(words) < 80 else 0)
    if missing_info > 50 and "Job description is suspiciously brief" not in reasons:
        reasons.append("Job description is suspiciously brief")

    # Email / Website Legitimacy
    has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+', text))
    has_url = bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
    company_kw = ["company", "inc", "llc", "ltd", "about us"]
    has_company_info = any(kw in text_lower for kw in company_kw)
    
    website_trust = 100 if has_url else 50
    email_authenticity = 100 if has_email else 50
    company_legitimacy = 100 if has_company_info else 40
    
    # Overall Scam Probability (heuristic combination)
    scam_prob = (base_risk * 0.4) + (urgency_score * 0.15) + (salary_score * 0.15) + ((100 - contact_authenticity) * 0.1) + ((100 - grammar_quality) * 0.1) + (missing_info * 0.1)
    scam_prob = min(scam_prob, 100.0)
    
    return {
        "Scam Probability": round(scam_prob, 1),
        "Urgency Manipulation": urgency_score,
        "Salary Manipulation": salary_score,
        "Grammar Quality": grammar_quality,
        "Contact Authenticity": contact_authenticity,
        "Website Trust": website_trust,
        "Email Authenticity": email_authenticity,
        "Company Legitimacy": company_legitimacy,
        "Missing Information": missing_info,
        "Final Risk Score": round(scam_prob, 1),
        "Reasons": reasons
    }

def calculate_risk(text: str):
    # Backward compatibility for single file analyzer
    stats = calculate_granular_risk(text)
    return stats["Final Risk Score"], stats["Reasons"]
