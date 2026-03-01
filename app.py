import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os
import numpy as np
import pandas as pd

# Set page config
st.set_page_config(page_title="Zomato Review Sentiment Analysis", page_icon="🍔", layout="centered")

# Custom CSS for a professional look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    .sentiment-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 10px;
    }
    .positive { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .neutral { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .negative { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

# Load assets
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "sentiment_model.keras")
    tokenizer_path = os.path.join(base_dir, "tokenizer.pkl")
    le_path = os.path.join(base_dir, "label_encoder.pkl")
    
    if not os.path.exists(model_path):
        return None, None, None
        
    model = tf.keras.models.load_model(model_path)
    
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    with open(le_path, 'rb') as handle:
        label_encoder = pickle.load(handle)
        
    return model, tokenizer, label_encoder

@st.cache_data
def load_random_reviews():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "Ratings.csv")
    if os.path.exists(dataset_path):
        # Sample directly from the first 50k rows for performance
        df = pd.read_csv(dataset_path, nrows=50000)
        df.dropna(subset=['review'], inplace=True)
        return df['review'].tolist()
    return ["The food was absolutely amazing!", "It was okay, nothing special.", "Worst experience ever."]

# Text cleaning function (Matches script_code.py)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# UI Header
st.title("🍔 Zomato Review Sentiment Analysis")
st.markdown("Instantly predict the sentiment of restaurant reviews using Deep Learning (LSTM).")

try:
    model, tokenizer, label_encoder = load_assets()
    
    if model is None:
        st.warning("⚠️ Model artifacts not found. Please run `python script_code.py` first to train the model.")
    else:
        # Session state for random review
        if 'review_input' not in st.session_state:
            st.session_state.review_input = ""

        # Layout for Input and Random Button
        col1, col2 = st.columns([4, 1])
        with col2:
            st.write("") # Spacer
            if st.button("🎲 Random"):
                reviews = load_random_reviews()
                st.session_state.review_input = np.random.choice(reviews)
                st.rerun()

        with col1:
            user_review = st.text_area("Enter your review:", value=st.session_state.review_input, placeholder="Type your experience here...")
            st.session_state.review_input = user_review

        if st.button("Predict Sentiment"):
            if user_review.strip():
                # Preprocess
                cleaned = clean_text(user_review)
                seq = tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=200) # Matches MAX_LEN in script_code.py
                
                # Predict
                prediction = model.predict(padded, verbose=0)
                sentiment_idx = np.argmax(prediction[0])
                confidence = prediction[0][sentiment_idx]
                label = label_encoder.inverse_transform([sentiment_idx])[0]
                
                # Display Result
                st.divider()
                
                # Dynamic Card
                card_class = label.lower()
                st.markdown(f"""
                <div class="sentiment-card {card_class}">
                    <h2 style="margin:0;">{label}</h2>
                    <p style="margin:0; font-size:1.1em;">Confidence: {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Specific messages
                if label == 'Positive':
                    st.balloons()
                    st.success("Great! This review shows high satisfaction.")
                elif label == 'Neutral':
                    st.info("The sentiment is neutral or mixed.")
                else:
                    st.error("This appears to be a negative or critical review.")
            else:
                st.warning("Please enter some text before predicting.")

except Exception as e:
    st.error(f"Something went wrong while loading the model. Ensure `script_code.py` has run successfully.\n\nDetails: {e}")

st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info("Model: Bidirectional LSTM\n\nClasses: Positive, Neutral, Negative\n\nDataset: Zomato Bangalore Reviews")
st.sidebar.write("---")
st.sidebar.markdown("### 🚀 How to Run")
st.sidebar.code("streamlit run app.py")
