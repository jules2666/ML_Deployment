
# ML Model Deployment with Docker

This project demonstrates how to deploy a machine learning model using Flask and Docker.

## 🔧 Setup Instructions

1. Clone the repo  
2. Train the model:
```bash
python app/train.py
```

3. Build and run Docker:
```bash
docker build -t ml-model .
docker run -p 9000:9000 ml-model
```

## ✅ API Endpoints

### `POST /predict`
**Input:**
```json
{
  "features": [[5.1, 3.5, 1.4, 0.2]]
}
```

**Output:**
```json
{
  "prediction": 0,
  "confidence": 0.97
}
```

### `GET /health`
```json
{
  "status": "ok"
}
```

## 🐳 Dockerized
The app runs inside a Docker container with Flask and scikit-learn.
