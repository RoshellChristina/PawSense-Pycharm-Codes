# PawSense ML Backend

This repository contains the backend APIs for the PawSense project, including Dog Emotion Classification, Disease Prediction, Wellness Score Prediction, and Text Embedding Service.

To set up the project, first clone the repository and navigate into the folder:

git clone <your-repo-url>
cd <repo-folder>

It is recommended to use a virtual environment:

python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows
pip install -r requirements.txt

Download the trained models and necessary artifacts from Google Drive and place them in the following structure:

project-root/
│
├─ dog_emotion_classifier.h5
├─ class_indices.json
├─ disease_prediction/
│   ├─ final_model.h5
│   ├─ preprocessor.pkl
│   ├─ label_encoder.pkl
│   └─ symptoms.json
├─ pipeline.pkl

Ensure the folder names match exactly for the Flask apps to locate the files correctly.

To run the Dog Emotion + Disease + Wellness API, execute:

python predict_app.py

The API will run on http://0.0.0.0:5000 with the following endpoints:
POST /predict — Dog emotion classification
POST /predict_disease — Disease prediction
POST /predict_score — Wellness score prediction

To run the Text Embedding Service, execute:

python embed_server.py

The API will run on http://0.0.0.0:6000 with the following endpoints:
POST /embed — Generate embeddings from text
GET /health — Health check

Make sure all required files from Google Drive are downloaded before running the APIs. Using a virtual environment is recommended to avoid dependency conflicts.
