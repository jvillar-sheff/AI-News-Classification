import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
import pandas as pd # <-- FIXED: Moved pandas import to the top

# --- Configuration ---
# FIXED: Updated path based on your directory structure
MODEL_DIR = "./models/Distilbert/distilbert-final"
MAX_LENGTH = 128
CLASS_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

@st.cache_resource
def load_classification_model(model_path):
    """Loads the fine-tuned DistilBERT model and tokenizer."""
    if not os.path.exists(model_path):
        st.error(f"Model directory not found at {model_path}.")
        st.stop()

    try:
        # Check for GPU and set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True
        ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )

        model.eval()
        st.success(f"Model loaded successfully on {device}!")
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        st.stop()
        
def classify_news(text, model, tokenizer, device):
    """Tokenizes text, runs inference, and returns predicted class probabilities."""
    
    # 1. Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    ).to(device)

    # 2. Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 3. Process logits to probabilities
    logits = outputs.logits
    # Ensure correct squeeze operation for batch size 1
    probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    # 4. Map probabilities to class names
    results = {}
    for i, prob in enumerate(probabilities):
        results[CLASS_NAMES[i]] = float(prob)
        
    return results

# --- Streamlit Application Layout ---
def main():
    st.set_page_config(
        page_title="DistilBERT News Classifier",
        layout="centered"
    )
    
    st.title("📰 DistilBERT News Classifier")
    st.markdown("A fine-tuned **DistilBERT** model to classify news articles into one of four categories: World, Sports, Business, or Sci/Tech.")

    # Load Model (cached)
    model, tokenizer, device = load_classification_model(MODEL_DIR)

    # User Input Area
    input_text = st.text_area(
        "Enter a News Article or Snippet:",
        placeholder="E.g., 'Google's new quantum computer achieved a massive breakthrough in computational speed, potentially revolutionizing drug discovery.'",
        height=200
    )

    if st.button("Classify Article"):
        if input_text:
            st.subheader("Classification Results")
            # Run prediction
            with st.spinner('Analyzing text...'):
                try:
                    probabilities = classify_news(input_text, model, tokenizer, device)
                    
                    # Sort results for display
                    sorted_probs = sorted(
                        probabilities.items(), 
                        key=lambda item: item[1], 
                        reverse=True
                    )
                    
                    # Display the top prediction
                    top_class = sorted_probs[0][0]
                    top_prob = sorted_probs[0][1] * 100
                    st.metric(label="Predicted Category", value=top_class, delta=f"{top_prob:.2f}% Confidence")
                    
                    # Display detailed breakdown
                    st.markdown("---")
                    st.text("Probability Breakdown:")
                    
                    # Prepare data for bar chart
                    labels = [item[0] for item in sorted_probs]
                    scores = [item[1] for item in sorted_probs]
                    
                    # Simple bar chart (Now works because pd is imported at the top)
                    prob_df = pd.DataFrame(scores, index=labels, columns=['Probability'])
                    st.bar_chart(prob_df, color="#1967D2")

                except Exception as e:
                    st.error(f"An error occurred during classification: {e}")
        else:
            st.warning("Please enter some text to classify.")

    st.markdown("---")
    st.caption("Model: Fine-tuned DistilBERT on AG News Dataset. Accuracy: 88.80%")
    
    # REMOVED the unnecessary local pandas import block
    # try:
    #     import pandas as pd
    # except ImportError:
    #     pass 

if __name__ == "__main__":
    main()