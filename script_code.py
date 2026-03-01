# ===============================
# 1. IMPORT LIBRARIES
# ===============================

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle

# Ensure prints are flushed immediately for visibility in logs
def log(msg):
    print(msg, flush=True)

# Define constants globally for use in functions
MAX_WORDS = 25000  # Increased for richer vocabulary
MAX_LEN = 200      # Increased to capture more context in long reviews

def run_training_pipeline():
    log("Starting Sentiment Analysis Training Pipeline...")

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import classification_report, confusion_matrix
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        log("TensorFlow and Scikit-learn libraries imported successfully.")
    except ImportError as e:
        log(f"Required libraries missing: {e}")
        log("Please run: pip install pandas numpy scikit-learn tensorflow matplotlib seaborn")
        sys.exit(1)

    # ===============================
    # 2. LOAD DATASET
    # ===============================

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "Ratings.csv")

    if not os.path.exists(file_path):
        log(f"CRITICAL ERROR: {file_path} not found.")
        sys.exit(1)

    log(f"Loading dataset from: {file_path}")
    # Loading 200k rows to ensure a diverse and representative sample for 3 classes
    df_full = pd.read_csv(file_path, nrows=200000) 
    log(f"Initial load complete. Rows: {len(df_full)}")

    # Drop rows with missing reviews or ratings
    df_full.dropna(subset=['review', 'rating'], inplace=True)

    # Sampling 120k rows for training (balanced if possible, but random is a good start)
    df = df_full.sample(n=min(120000, len(df_full)), random_state=42)
    log(f"Cleaning and processing {len(df)} sampled reviews...")

    # ===============================
    # 3. DEFINE SENTIMENT (3 CLASSES)
    # ===============================

    def define_sentiment(rating):
        try:
            r = float(rating)
            if r >= 4.0:
                return 'Positive'
            elif r >= 3.0:
                return 'Neutral'
            else:
                return 'Negative'
        except:
            return 'Neutral'

    df['Sentiment'] = df['rating'].apply(define_sentiment)
    df = df[['review', 'Sentiment']]

    log("\n--- Class Distribution ---")
    log(df['Sentiment'].value_counts())

    # ===============================
    # 4. TEXT CLEANING
    # ===============================

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters but keep punctuation that might indicate sentiment (e.g., !) is tricky,
        # but for LSTM, standardizing to letters is generally safer.
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    log("Performing text normalization...")
    df['review'] = df['review'].apply(clean_text)

    # ===============================
    # 5. ENCODE LABELS
    # ===============================

    le = LabelEncoder()
    df['Sentiment_Encoded'] = le.fit_transform(df['Sentiment'])
    log(f"Target Labels: {list(le.classes_)}")
    log(f"Label Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ===============================
    # 6. TOKENIZATION & PADDING
    # ===============================

    log(f"Tokenizing vocabulary (Max words: {MAX_WORDS})...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['review'])

    sequences = tokenizer.texts_to_sequences(df['review'])
    X = pad_sequences(sequences, maxlen=MAX_LEN)
    y = df['Sentiment_Encoded'].values

    # ===============================
    # 7. TRAIN TEST SPLIT
    # ===============================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # ===============================
    # 8. BUILD ADVANCED LSTM MODEL
    # ===============================

    log("Constructing Deep Bidirectional LSTM Architecture...")
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
        SpatialDropout1D(0.4),  # Regularization to prevent overfitting on sequences
        
        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Bidirectional(LSTM(32)),
        BatchNormalization(),
        
        Dropout(0.5), # High dropout for robustness
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(3, activation='softmax') # Multi-class output
    ])

    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )

    model.summary()

    # ===============================
    # 9. TRAIN MODEL WITH OPTIMIZED CALLBACKS
    # ===============================

    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True, 
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=2, 
        min_lr=0.00001, 
        verbose=1
    )

    log("Training model (this could take 5-10 mins depending on CPU/GPU)...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.15,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # ===============================
    # 10. EVALUATION & METRICS
    # ===============================

    log("\nEvaluating model on unseen test data...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    log(f"Final Test Accuracy: {accuracy:.4f}")

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    log("\n--- Classification Report ---")
    log(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save visualization
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy Progress')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "training_metrics.png"))
    log("Training visualization saved to training_metrics.png")

    # ===============================
    # 11. SAVE ARTIFACTS FOR DEPLOYMENT
    # ===============================
    log("\nPackaging model and preprocessors for Streamlit...")

    # Save Keras Model
    model_path = os.path.join(base_dir, "sentiment_model.keras")
    model.save(model_path)
    log(f"Saved Model: {model_path}")

    # Save Tokenizer (Pickle format as requested)
    tokenizer_path = os.path.join(base_dir, "tokenizer.pkl")
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"Saved Tokenizer: {tokenizer_path}")

    # Save LabelEncoder
    label_encoder_path = os.path.join(base_dir, "label_encoder.pkl")
    with open(label_encoder_path, 'wb') as handle:
        pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"Saved LabelEncoder: {label_encoder_path}")

    log("\nPipeline completed. You can now run the Streamlit app using: streamlit run app.py")

if __name__ == "__main__":
    run_training_pipeline()
