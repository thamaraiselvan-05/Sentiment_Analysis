import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Set MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Sentiment_Analysis")

# Load data
df = pd.read_csv("cleaned_data.csv")

# Remove empty reviews
df = df.dropna(subset=['clean_review'])
df = df[df['clean_review'].str.strip() != ""]

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review']).toarray()
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Try multiple hyperparameters
for C_value in [0.1, 1.0, 10.0]:
    with mlflow.start_run(run_name=f"LogReg_C_{C_value}"):

        model = LogisticRegression(C=C_value, max_iter=200)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)

        print(f"C={C_value} | F1={f1}")

        # Log parameters and metrics
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C_value)
        mlflow.log_metric("f1_score", f1)

        # Save locally
        joblib.dump(model, "sentiment_model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")

        # Log artifacts
        mlflow.log_artifact("sentiment_model.pkl")
        mlflow.log_artifact("vectorizer.pkl")

        # Register model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="SentimentClassifier"
        )

print("Training complete with MLflow tracking!")
