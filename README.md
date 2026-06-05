# MLOps Blog - Part 2: From Experiments to Production Deployment

This repository contains the code for **Part 2** of the MLOps Blog series.

## What This Part Covers

Part 2 builds on Part 1 and introduces the following major concepts:

### 1. MLflow Experiment Tracking (Week 5)
- Running multiple experiments with different hyperparameters
- Tracking experiments, metrics, and models using MLflow
- Comparing experiment results

### 2. Model Serving with FastAPI (Week 6-7)
- Building a production-grade REST API using FastAPI
- Loading models from MLflow Model Registry
- Structured logging and request tracing

### 3. Containerization with Docker (Week 6-7)
- Dockerizing the application
- Using Docker Compose for local development

### 4. Kubernetes Deployment (Week 6-7)
- Deploying the model serving API on Kubernetes
- Horizontal Pod Autoscaler (HPA) for auto-scaling
- Production-ready deployment manifests

## Project Structure

```
MLOps-Blog-Part-2/
├── README.md
├── main.py                    # Pipeline orchestrator with MLflow
├── app.py                     # FastAPI model serving application
├── run_experiments.py         # MLflow experiment runner
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
└── src/
    ├── data_processing.py
    ├── model_training.py      # MLflow integrated version
    └── dvc_operations.py
```

## How to Use

### 1. Run Experiments with MLflow

```bash
python run_experiments.py --data-path iris-dvc-pipeline/v1_data.csv
```

Then view results:
```bash
mlflow ui
```

### 2. Run the API Locally

```bash
python app.py
```

Or using Docker:
```bash
docker-compose up --build
```

### 3. Deploy to Kubernetes

```bash
kubectl apply -f kubernetes/
```

## Progression from Part 1

| Part | Focus                              | Key Technologies                     |
|------|------------------------------------|--------------------------------------|
| 1    | Reliable ML Pipeline               | DVC, Testing, CI/CD, CML            |
| 2    | Experiments + Model Serving        | **MLflow, FastAPI, Docker, Kubernetes** |

## Author

Vaibhav Gupta

Part of the MLOps Blog Series.
