# Real Estate Price Predictor

> **Full-Stack ML Web App for Bangalore Real Estate Price Prediction**

## 🛠️ Tech Stack & Tools

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Angular](https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white)](https://angular.io/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com/)
[![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)
[![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

---

## Project Overview

The application allows users to estimate the price of a house in Bangalore based on features such as area (in square feet), number of bedrooms (BHK), number of bathrooms, and location. The prediction is powered by a machine learning model trained on real-world housing data.

---

## Machine Learning Model
- **Type:** Supervised regression (Linear Regression)
- **Features:**
  - Total square footage
  - Number of bathrooms
  - Number of bedrooms (BHK)
  - Location (one-hot encoded)
- **Training Data:**
  - Sourced from a comprehensive dataset of Bangalore house prices
  - Extensive preprocessing: outlier removal, feature engineering, categorical encoding
- **Artifacts:**
  - Trained model serialized as `banglore_home_prices_model.pickle`
  - Feature/column metadata in `columns.json`

---

## Backend: Flask REST API
- **Framework:** Python Flask
- **Endpoints:**
  - `GET /get_location_names` — Returns all available locations for the dropdown
  - `POST /predict_home_price` — Accepts form data and returns the predicted price
- **Model Loading:**
  - Loads the pickled model and column metadata at startup (or on first request)
- **Prediction Flow:**
  - Receives user input, constructs a feature vector/DataFrame, and returns the model's price prediction
- **CORS:**
  - Configured to allow cross-origin requests from the frontend (for local/dev and production)
- **Containerization:**
  - Fully Dockerized for easy deployment (Dockerfile, docker-compose)

---

## Frontend: Angular + Tailwind CSS
- **Framework:** Angular (standalone components, modern architecture)
- **Styling:** Tailwind CSS for utility-first, responsive, and beautiful UI
- **Features:**
  - Dynamic form for area, BHK, bathrooms, and location
  - Real-time fetching of available locations from the backend
  - Displays predicted price with smooth UI feedback and error handling
  - Responsive, glassmorphism-inspired design for modern look and feel
- **API Integration:**
  - Uses Angular's HttpClient to communicate with the Flask backend
  - Handles loading states, errors, and result display
- **Deployment Ready:**
  - Can be deployed as a static site (Vercel, Netlify, etc.) or with SSR if needed

---

## Architecture Diagram

```
User (Browser)
   │
   ▼
Angular + Tailwind Frontend (Vercel/Static Hosting)
   │  (HTTP/REST)
   ▼
Flask REST API (Docker/Render/Cloud)
   │
   ▼
ML Model (Pickle) + Feature Metadata (JSON)
```

---

## Technical Highlights
- **End-to-end ML pipeline:** Data cleaning, feature engineering, model training, serialization
- **API-first backend:** Clean separation of model logic and HTTP interface
- **Modern frontend:** Angular standalone components, Tailwind for rapid UI development
- **Production-ready:** Dockerized backend, CORS, environment-agnostic frontend
- **Extensible:** Easy to swap out the ML model, add new features, or adapt to other cities

---

## Authors & Credits
- ML, backend, and frontend: [Your Name/Team]
- Data: Bangalore housing dataset (public sources)

---

For more details on the ML pipeline, API contract, or frontend architecture, see the code and comments in each respective directory.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Repository Contents](#repository-contents)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/real-estate-price-prediction.git
    ```
2. Navigate to the project directory:
    ```sh
    cd real-estate-price-prediction
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Jupyter Notebook:
    ```sh
    jupyter notebook Real\ Estate\ Price\ Prediction.ipynb
    ```
2. Follow the instructions in the notebook to train the model and make predictions.

## Features

- Data preprocessing
- Model training
- Model evaluation
- Price prediction

## Repository Contents

| File/Folder                  | Description                                      |
|------------------------------|--------------------------------------------------|
| `Real Estate Price Prediction.ipynb` | Jupyter Notebook for the project.               |
| `Real Estate Price Prediction.py`   | Python script for the project.                  |
| `columns.json`               | JSON file containing column names.               |
| `requirements.txt`           | List of dependencies required for the project.   |
| `README.md`                  | This README file.                                |

## Code Excerpt

```python
import json
columns={
    'data_columns':[col.lower() for col in x.columns]
}
with open ('columns.json','w') as f:
    f.write(json.dumps(columns))
