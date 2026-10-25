"""Train the demo crop classifier using the included synthetic dataset."""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "crop_demo_data.csv"
MODEL_PATH = BASE_DIR / "crop_model.joblib"

df = pd.read_csv(DATA_PATH)
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=180, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Demo holdout accuracy: {accuracy_score(y_test, predictions):.3f}")
print(classification_report(y_test, predictions, zero_division=0))

# Refit on all included examples for the shipped demo model.
model.fit(X, y)
joblib.dump(model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
