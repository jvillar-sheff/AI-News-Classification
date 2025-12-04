import streamlit as st
import torch
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from huggingface_hub import hf_hub_download
import os

# --- 1. CONFIGURATION ---
ADAPTER_REPO = "jvillar-sheff/ag-news-distilbert-lora"
BASE_MODEL_ID = "distilbert-base-uncased"
CLASS_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# --- 2. PAGE SETUP ---
st.set_page_config(page_title="News Classifier", page_icon="📰", layout="centered")

# --- 3. DYNAMIC METRICS LOADING ---
@st.cache_data(ttl=3600)  # Cache for 1 hour so we don't download every second
def fetch_metrics():
    """Downloads evaluation_report.json from the Model Hub."""
    try:
        file_path = hf_hub_download(repo_id=ADAPTER_REPO, filename="evaluation_report.json")
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Extract numbers (assuming your JSON structure from the notebook)
        # Adjust keys if your specific JSON structure differs slightly
        acc = data['overall_metrics']['Accuracy']
        f1 = data['overall_metrics']['F1 Macro']
        
        return {
            "Accuracy": f"{acc:.2%}",
            "F1_Score": f"{f1:.4f}"
        }
    except Exception as e:
        # Fallback if file missing
        return {"Accuracy": "N/A", "F1_Score": "N/A"}

# Load metrics immediately
MODEL_METRICS = fetch_metrics()

# --- 4. MODEL LOADING (Cached) ---
@st.cache_resource
def load_model():
    try:
        # Load Base Model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_ID,
            num_labels=len(CLASS_NAMES),
            id2label={k: v for k, v in enumerate(CLASS_NAMES.values())},
            label2id={v: k for k, v in CLASS_NAMES.items()}
        )
        
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)
        
        # Load LoRA Adapters
        model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
        
        # Force CPU (Standard for free Hugging Face Spaces)
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, tokenizer, device = load_model()

# --- 5. PREDICTION FUNCTION ---
def predict(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    pred_idx = np.argmax(probs)
    pred_label = CLASS_NAMES[pred_idx]
    pred_conf = probs[pred_idx]
    
    class_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    
    return pred_label, pred_conf, class_probs

# --- 6. USER INTERFACE ---

st.title("📰 NLP News Classifier")
st.markdown("""
This interface uses a **DistilBERT** model fine-tuned with **LoRA (Low-Rank Adaptation)**.
It classifies news headlines into four categories: **World, Sports, Business, or Sci/Tech**.
""")

# Dynamic Green Banner
st.success(f"✅ **Model Performance:** Accuracy: {MODEL_METRICS['Accuracy']} | F1 Score: {MODEL_METRICS['F1_Score']}")

text_input = st.text_area(
    "Enter a News Article or Snippet:", 
    height=150, 
    placeholder="e.g., The stock market rallied today as tech companies reported record profits..."
)

if st.button("Classify Article", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence, all_probs = predict(text_input)
            
        st.divider()
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.subheader("Prediction")
            st.markdown(f"<h1>{label}</h1>", unsafe_allow_html=True)
            
            # Stylized Badge
            if confidence > 0.85:
                bg_color, text_color, icon = "#1b4332", "#4ade80", "↑"
            elif confidence > 0.60:
                bg_color, text_color, icon = "#453a1f", "#facc15", "~"
            else:
                bg_color, text_color, icon = "#451a1a", "#f87171", "↓"

            st.markdown(
                f"""
                <div style="
                    background-color: {bg_color}; color: {text_color};
                    padding: 5px 15px; border-radius: 20px; display: inline-flex;
                    align-items: center; font-weight: 600; font-size: 14px;
                    border: 1px solid {text_color}40; margin-bottom: 10px;">
                    {icon} &nbsp; {confidence:.2%} Confidence
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col2:
            st.subheader("Probability Breakdown")
            df_probs = pd.DataFrame(list(all_probs.items()), columns=['Category', 'Probability'])
            st.bar_chart(df_probs.set_index('Category'))

st.markdown("---")
st.caption("Built by Joaquin Villar Urrutia | Powered by Hugging Face & Streamlit")