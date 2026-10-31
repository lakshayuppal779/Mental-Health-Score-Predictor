# Mental Health Score Predictor

Predict a student's **mental health score (0–10)** from their social media habits, sleep, study time, physical activity and stress level, using a machine learning model served through a FastAPI backend and a clean web frontend.

**Live demo:** https://mental-health-score-predictor-1-15m4.onrender.com

> The backend runs on Render's free tier, so it sleeps when idle. The first prediction after a break can take 30–60 seconds. Later ones are instant.

> **Disclaimer:** This project is for learning and informational purposes only. It is **not** a clinical or diagnostic tool. If you are struggling, please talk to someone you trust or a mental health professional.

---

## Features

- Regression model trained on 5,000 student records
- REST API built with **FastAPI** with strict input validation (Pydantic)
- Responsive frontend with form validation and an animated score gauge
- Friendly result bands: *strained*, *balanced*, *strong*
- Deployed on **Render** (backend and frontend)

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data and ML | Python, pandas, NumPy, scikit-learn, seaborn, matplotlib |
| Backend | FastAPI, Uvicorn, Pydantic, joblib |
| Frontend | HTML, CSS, vanilla JavaScript |
| Deployment | Render, GitHub |

## Project Structure

```
Mental-Health-Score-Predictor/
├── backend/
│   ├── main.py                    # FastAPI app with the /predict endpoint
│   ├── Mental_Health_Model.pkl    # Trained scikit-learn pipeline
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── script.js                  # Form handling and API calls
│   └── style.css
├── notebook/
│   ├── ML_Project.ipynb           # EDA, preprocessing, training, evaluation
│   └── Student_Social_Media_And_Mental_Health_Impact.csv
├── .gitignore
└── README.md
```

## The Model

**Problem:** regression, predicting `Mental_Health_Score` (a continuous value between roughly 3 and 10).

**Input features (12):** Age, Gender, Country, Academic Level, Most Used Platform, Purpose of Use, Avg Daily Usage Hours, Daily Unlocks, Study Hours, Physical Activity Hours, Sleep Hours Per Night, Stress Level.

**Pipeline**
1. Removed duplicates and clipped unrealistic negative values in physical activity.
2. Grouped rare countries into an `Other` category (top 10 kept).
3. Preprocessing with a `ColumnTransformer`:
   - Skewed feature (`Study_Hours`): log transform + standard scaling
   - Other numeric features: standard scaling
   - Stress level: ordinal encoding (Low < Medium < High < Very High)
   - Categorical features: one-hot encoding
4. Trained and compared Linear Regression and Random Forest, with a train/test split of 70/30.

**Results (test set)**

| Model | R² | MAE | RMSE |
|-------|-----|------|-------|
| Linear Regression | 0.740 | 0.536 | 0.676 |
| **Random Forest (default), used in the app** | **0.878** | **0.347** | **0.464** |
| Random Forest (tuned with RandomizedSearchCV) | 0.865 | 0.369 | 0.487 |

The default Random Forest gave the best test performance and is the model saved in `Mental_Health_Model.pkl`.

## API

**Base URL:** `https://mental-health-score-predictor-k96n.onrender.com`
Interactive docs: `/docs`

### `POST /predict`

Request body:

```json
{
  "age": 21,
  "gender": "Male",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 5,
  "daily_unlocks": 60,
  "study_hours": 4,
  "physical_activity_hours": 1,
  "sleep_hours_per_night": 7,
  "stress_level": "Medium"
}
```

Response:

```json
{
  "predicted_mental_health_score": 6.43
}
```

**Allowed values**

| Field | Values / range |
|-------|----------------|
| `age` | integer, 10–100 |
| `gender` | `Male`, `Female` |
| `academic_level` | `High School`, `Undergraduate`, `Graduate` |
| `most_used_platform` | Facebook, Instagram, Snapchat, Twitter, YouTube, TikTok, LinkedIn, LINE, KakaoTalk, VKontakte, WhatsApp, WeChat |
| `purpose_of_use` | `Networking`, `Education`, `Entertainment`, `News` |
| `avg_daily_usage_hours`, `study_hours`, `physical_activity_hours`, `sleep_hours_per_night` | 0–24 |
| `daily_unlocks` | integer, 0 or more |
| `stress_level` | `Low`, `Medium`, `High`, `Very High` |
| `country` | any text (unlisted countries are grouped as `Other`) |

## Run Locally

### Prerequisites
- Python 3.12
- Git

### 1. Clone the repository
```bash
git clone https://github.com/lakshayuppal779/Mental-Health-Score-Predictor.git
cd Mental-Health-Score-Predictor
```

### 2. Set up the backend
```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt

cd backend
uvicorn main:app --port 2200 --reload
```
Open http://127.0.0.1:2200/docs to try the API.

> Use the same scikit-learn version that is pinned in `requirements.txt`. A different version may fail to load the `.pkl` model.

### 3. Run the frontend
In `frontend/script.js`, point the API to your local server:
```js
const API_BASE = "http://127.0.0.1:2200";
```
Then, in a second terminal:
```bash
cd frontend
python3 -m http.server 5500
```
Open http://localhost:5500.

## Deployment

- **Backend:** Render Web Service, root directory `backend`, build command `pip install -r requirements.txt`, start command `uvicorn main:app --host 0.0.0.0 --port $PORT`, with `PYTHON_VERSION=3.12.3`.
- **Frontend:** Render Static Site, root directory `frontend`, publish directory `.`.

## Dataset

Student Social Media and Mental Health Impact: 5,000 students and 13 columns covering demographics, social media usage, lifestyle habits and a mental health score.

## Future Improvements

- Show which factors influenced the score (feature importance)
- Add model monitoring and more robust hyperparameter tuning
- Add unit tests for the API

## Author

**Lakshay Uppal**
GitHub: [@lakshayuppal779](https://github.com/lakshayuppal779)
