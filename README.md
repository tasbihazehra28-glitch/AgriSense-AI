# AgriSense AI 🌱
### AI-powered crop recommendation prototype

AgriSense AI is a small hackathon-ready demonstration of how machine learning can help organize basic soil and climate information into a crop suggestion. It uses a Random Forest classifier and a simple Streamlit web interface.

## Features
- Crop suggestion from nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, soil pH, and rainfall.
- Model confidence score.
- Basic pH guidance and reminders to validate recommendations locally.
- Included demo dataset and training script.
- Interactive dataset preview in the app.

## Important limitation
**The included dataset is synthetic demo data, not a validated agricultural dataset.** The model demonstrates a machine-learning workflow and must not be used to make real planting decisions without validated regional data and advice from agricultural experts. Its confidence score is not a probability of crop success or yield.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

The browser should open the app automatically.

## Retrain the model

The trained model (`crop_model.joblib`) is included so the app can run immediately. To retrain it from the included dataset:

```bash
python train_model.py
```

## Files
- `app.py` — Streamlit user interface and prediction flow
- `train_model.py` — model training and evaluation script
- `crop_model.joblib` — trained demo model
- `data/crop_demo_data.csv` — synthetic sample dataset
- `requirements.txt` — Python dependencies

## Suggested next steps for a real-world version
1. Replace the synthetic examples with a trusted, region-specific dataset.
2. Validate the model with agronomists and independent test data.
3. Add location-aware seasonal and water availability inputs.
4. Add multilingual support and explainable recommendations.
5. Evaluate performance by crop and region before deployment.

## Hackathon presentation pitch
"AgriSense AI demonstrates how machine learning can turn soil and climate inputs into an understandable crop suggestion. The prototype is designed as a starting point for data-driven agriculture, with a clear path toward validated regional data, explainability, and farmer-friendly access."
