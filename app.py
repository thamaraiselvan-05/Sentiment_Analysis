import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
st.set_page_config(layout="centered")

# Download resources
nltk.download('stopwords')
nltk.download('wordnet')

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# ------------------ PAGE DESIGN ------------------

st.markdown("""
<style>

/* 🌈 MAIN APP BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f3f7ff, #e6f0ff, #f9f0ff);
}

/* Force card width */
.block-container {
    max-width: 700px;
    margin: auto;
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

/* 🫧 Soft Glow Blobs */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: -120px;
    left: -120px;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, #d6e4ff, transparent);
    z-index: 0;
}

[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    bottom: -120px;
    right: -120px;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, #f3d1ff, transparent);
    z-index: 0;
}

/* 💎 Glass Card Effect */
[data-testid="stVerticalBlock"] {
    background: rgba(255, 255, 255, 0.75);
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

/* 🎯 Title */
h1 {
    text-align: center;
    color: #2d3a5a;
    font-weight: 600;
}

/* ✍️ Text Area */
textarea {
    border-radius: 14px !important;
    border: 1px solid #d0d7e2 !important;
    background: #ffffffcc !important;
}

/* 🔵 Button */
.stButton>button {
    background: linear-gradient(90deg, #6fb1fc, #4364f7);
    color: white;
    border-radius: 14px;
    padding: 12px 28px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.05);
    transition: 0.3s;
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# ------------------ APP UI ------------------

st.markdown("<h1>🛒 Flipkart Review Sentiment Analyzer</h1>", unsafe_allow_html=True)

st.write("💬 Enter a product review below and let AI predict the sentiment!")

review = st.text_area("✍️ Your Review Here", height=150)

if st.button("🔍 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        clean = clean_text(review)
        vec = vectorizer.transform([clean])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("✨ Positive Review — Customers are happy!")
            st.balloons()
        else:
            st.error("⚠ Negative Review — Improvement needed!")
