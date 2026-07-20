import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.file_parser import parse_file
from utils.risk_calculator import calculate_granular_risk
from services.model_service import load_model
import streamlit as st

def process_batch(uploaded_files, progress_callback=None):
    """
    Process a batch of uploaded files.
    uploaded_files: list of Streamlit UploadedFile objects
    progress_callback: function(current, total)
    """
    model, vectorizer = load_model()
    
    results = []
    texts = []
    
    start_time = time.time()
    
    for i, file in enumerate(uploaded_files):
        if progress_callback:
            progress_callback(i + 1, len(uploaded_files))
            
        file_bytes = file.getvalue()
        text, err = parse_file(file.name, file_bytes)
        
        if err:
            results.append({
                "Filename": file.name,
                "Status": "Error",
                "Error": err,
                "Text": ""
            })
            texts.append("")
            continue
            
        if not text or len(text.strip()) == 0:
            results.append({
                "Filename": file.name,
                "Status": "Error",
                "Error": "No text extracted",
                "Text": ""
            })
            texts.append("")
            continue
            
        # Analysis
        cleaned = text.lower()
        vec_out = vectorizer.transform([cleaned])
        pred = model.predict(vec_out)[0]
        proba = model.predict_proba(vec_out)[0]
        fake_prob = proba[1] * 100
        
        risk_stats = calculate_granular_risk(text)
        
        # Combine ML probability and Heuristic Risk
        ai_confidence = fake_prob if pred == 1 else (proba[0] * 100)
        
        # Final Verdict
        # We classify as Fraudulent if ML pred is 1 OR heuristic risk > 60
        verdict = "Fake" if pred == 1 or risk_stats["Final Risk Score"] > 60 else "Legitimate"
        if verdict == "Legitimate" and risk_stats["Final Risk Score"] > 25:
            verdict = "Suspicious"
            
        risk_level = "High" if verdict == "Fake" else ("Medium" if verdict == "Suspicious" else "Low")
        
        results.append({
            "Filename": file.name,
            "Status": "Success",
            "Verdict": verdict,
            "Risk Level": risk_level,
            "Fraud Score": risk_stats["Final Risk Score"],
            "AI Confidence": round(ai_confidence, 1),
            "Risk Stats": risk_stats,
            "Text": text
        })
        texts.append(text)
        
    processing_time = time.time() - start_time
    
    # Duplicate Detection
    duplicates = detect_duplicates(texts, vectorizer)
    
    return results, duplicates, processing_time

def detect_duplicates(texts, vectorizer):
    """
    Detects duplicate or near-duplicate texts using TF-IDF cosine similarity.
    Returns a list of tuples (idx1, idx2, similarity_percentage)
    """
    if len(texts) < 2:
        return []
        
    # Filter out empty texts for duplicate detection
    valid_indices = [i for i, t in enumerate(texts) if t.strip()]
    if len(valid_indices) < 2:
        return []
        
    valid_texts = [texts[i] for i in valid_indices]
    vec_out = vectorizer.transform(valid_texts)
    sim_matrix = cosine_similarity(vec_out)
    
    duplicates = []
    for i in range(len(valid_indices)):
        for j in range(i + 1, len(valid_indices)):
            sim = sim_matrix[i][j] * 100
            if sim >= 85: # Threshold for near duplicate
                duplicates.append({
                    "doc1_idx": valid_indices[i],
                    "doc2_idx": valid_indices[j],
                    "similarity": round(sim, 1)
                })
                
    return duplicates
