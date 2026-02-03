import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load data
df = pd.read_csv("data.csv")

# Create sentiment column
df['sentiment'] = df['Ratings'].apply(lambda x: 1 if x >= 4 else 0)

# Text cleaning setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df['clean_review'] = df['Review text'].apply(clean_text)

df.to_csv("cleaned_data.csv", index=False)
print("✅ Data cleaned and saved!")
