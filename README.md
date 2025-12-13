# 🚕 Taxi Next-Node Prediction – End-to-End MLOps System

## Overview

This project implements an **end-to-end, production-ready ML system** for predicting the **next likely geographic node (geohash)** in a taxi trip, based on historical trajectory data.

Starting from **raw trip data**, the system covers the full MLOps lifecycle:
- Distributed data preprocessing
- Feature engineering
- Model training and evaluation
- Batch orchestration
- Model serving via HTTP API
- Monitoring and telemetry
- Testing and environment configuration (DTAP)

The focus of this project is **system design, correctness, and reproducibility**, rather than maximizing model performance.

---

## Problem Statement

Given a partial taxi trip defined by:
- Previous location (`prev_node`)
- Current location (`curr_node`)
- Time features (hour, day of week)
- Call type

the system predicts the **top-k most likely next locations** in the trip.

This is framed as a **multi-class classification problem**, where each class corresponds to a geohash node.

---

## Dataset

The project is based on the **Porto Taxi Trajectory Dataset** (UCI Machine Learning Repository).

- **Source**: UCI Machine Learning Archive  
- **Task type**: Classification  
- **Scale**: Large-scale trip-level dataset  
- **Raw format**: CSV with GPS polylines  

Each trip consists of a sequence of GPS coordinates, which are transformed into **geohash sequences** during preprocessing.

---

## System Architecture

The system is composed of the following stages:

1. **Data Preprocessing (Batch, Distributed)**
   - Raw CSV ingestion using **Dask**
   - Parsing GPS polylines
   - Conversion to geohash sequences
   - Trip filtering and time feature extraction
   - Output stored as Parquet

2. **Feature Engineering**
   - Conversion of trips into node-to-node transitions
   - Temporal feature encoding (cyclical time, semantic buckets)
   - Train/test split
   - Output stored as Parquet

3. **Model Training**
   - Baseline **Logistic Regression** classifier
   - Categorical + numerical feature pipeline (scikit-learn)
   - Model artefacts and metadata saved via Joblib

4. **Graph Construction**
   - Transition graph built from training data
   - Used to enrich API responses with graph-level statistics

5. **Batch Orchestration**
   - Training pipeline orchestrated using **Apache Airflow**
   - Fully containerized execution

6. **Model Serving**
   - **FastAPI**-based HTTP service
   - Predicts top-k next nodes
   - Exposes health, graph, and metrics endpoints

7. **Monitoring & Telemetry**
   - Metrics collected using **Prometheus**
   - Dashboards visualized in **Grafana**

---

## Tech Stack

### Core
- **Language**: Python 3.10
- **Distributed Processing**: Dask
- **ML Framework**: scikit-learn

### Orchestration & Deployment
- **Batch Pipeline**: Apache Airflow
- **API Framework**: FastAPI
- **Containerization**: Docker
- **Service Composition**: Docker Compose

### Monitoring
- **Metrics**: Prometheus
- **Dashboards**: Grafana
- **Alerting**: Prometheus AlertManager

### Testing
- **Testing Framework**: Pytest
- **Unit Tests**: Core data and feature logic
- **Integration Tests**: API contract testing
- **Test Environment**: Dedicated Docker Compose setup (`ENV=test`)

---

## Configuration & Environments (DTAP)

The system supports multiple environments through **configuration-driven design**.  
All paths, preprocessing parameters, feature settings, and training options are defined in **YAML configuration files**.

The active environment is selected via the `ENV` environment variable:

```bash
ENV=dev
ENV=test
ENV=prod
```

Configuration files are located under:

configs/
├── dev.yaml
├── test.yaml
└── prod.yaml

---

## Testing Strategy

The project follows a **layered testing strategy** designed to ensure correctness, reliability, and maintainability while keeping tests fast and deterministic.

### Unit Tests

Unit tests focus on **core business logic and data transformations**, including:
- Data preprocessing functions (parsing, geohash conversion, filtering)
- Feature engineering logic (trip-to-transition expansion, time encoding)
- Transition graph construction
- Training pipeline orchestration (lightweight validation only)

External systems and frameworks (e.g., Dask execution, scikit-learn internals, filesystem I/O) are mocked where appropriate to isolate logic under test.

### Integration Tests

Integration tests validate **system-level behaviour** without relying on production artefacts.  
These tests focus on:
- API endpoint wiring and routing
- Request and response schema validation
- Error handling behavior
- Dependency integration between API components

Model and graph dependencies are mocked to ensure tests remain fast, deterministic, and environment-independent.

### Test Environment

All tests are executed inside a **dedicated test container** using Docker Compose, with configuration isolated via:

```bash
ENV=test
```

### Test execution

Unit and integration tests can be executed using:

```bash
docker compose -f infra/docker-compose.test.yaml up --build --abort-on-container-exit
```
---

## Running the System

The system is composed of three main runtime components:
1. **Batch training pipeline (Airflow)**
2. **Model serving API (FastAPI)**
3. **Monitoring stack (Prometheus + Grafana)**

Each component is containerized and can be run independently using Docker Compose.

---

## Running the Training Pipeline (Airflow)

The batch training pipeline is orchestrated using **Apache Airflow**.  
It implements the full offline ML workflow, from raw data ingestion to trained model artefacts.

### Pipeline Overview

The Airflow DAG `taxi_next_node_training` consists of the following tasks:

1. **Preprocess Trips**
   - Loads raw CSV data
   - Parses GPS polylines
   - Converts coordinates to geohash sequences
   - Outputs preprocessed Parquet files

2. **Build Transition Features**
   - Converts trips into node-to-node transitions
   - Extracts temporal and semantic features
   - Splits data into train and test sets

3. **Build Transition Graph**
   - Builds a transition-count graph from training data
   - Saves the graph as a JSON artefact

4. **Train Model**
   - Trains a baseline logistic regression classifier
   - Evaluates accuracy and top-k accuracy
   - Saves the trained model bundle and metadata

The task dependency chain is:

```text
preprocess → build_features → build_graph → train
```

### Running Airflow

To start Airflow:

```bash
docker compose -f infra/docker-compose.airflow.yaml up --build
```

The Airflow UI will be available at:

```text
http://localhost:8080
```

with the DAG under the name **taxi_next_node_training**. The pipeline can be triggered manually from the UI.

---

## Running the Model Serving API
The trained model is served via a FastAPI-based HTTP service.

The API loads:

- The trained model bundle
- The transition graph artefact

at startup, based on the active environment configuration (intended as prod).

### Running the API

```bash
docker build -t taxi-api:latest -f infra/Dockerfile.api .
docker run \
  --name api \
  -p 8000:8000 \
  -e ENV=prod \
  -e PYTHONPATH=/app \
  taxi-api:latest
curl http://localhost:8000/health
```
The API will be available at 
```text
http://localhost:8000
```

## API Endpoints

The API exposes endpoints for health checking, prediction, graph inspection, and telemetry.

---

### `GET /health`

Returns basic service and model information.

**Response example:**
```json
{
  "status": "ok",
  "model_version": "1.0.0",
  "top_k": 3
}
```

### `POST /predict_next_node`
Predicts the top-k most likely next nodes in a taxi trip.

**Example Request:**
```bash
curl -X POST http://localhost:8000/predict_next_node \
  -H "Content-Type: application/json" \
  -d '{
    "prev_node": "ezjmgt",
    "curr_node": "ezjmgu",
    "hour": 14,
    "day_of_week": 2,
    "call_type": "A"
  }'
```

**Response example:**
```json
[
  {
    "node": "ezjmgs",
    "probability": 0.52
  },
  {
    "node": "ezjmgt",
    "probability": 0.31
  },
  {
    "node": "ezjmgu",
    "probability": 0.17
  }
]
```

### `GET /graph_node_info/{node}`
Returns transition graph statistics for a given node.

### `GET /metrics`
Exposes Prometheus-compatible metrics in plaintext format.

Metrics include:

- Request counters
- Error counters
- Latency histograms
- Model accuracy and top-k accuracy
- Prediction confidence and entropy

This endpoint is scraped by Prometheus and visualized in Grafana.

---

## Monitoring and Telemetry

The system includes a monitoring stack based on **Prometheus** and **Grafana** to provide observability into API availability, performance, and model behavior.

Monitoring is decoupled from the application logic and runs as an independent set of services.

### Prometheus

Prometheus scrapes the API metrics endpoint and stores time-series data for monitoring and alerting.

Prometheus is available at:

```text
http://localhost:9090
```

### Grafana

Grafana is used to visualize Prometheus metrics through pre-provisioned dashboards.

Grafana is available at:

```text
http://localhost:3000
```

### Running the Monitoring Stack

One can start all the monitoring services (including the API) via:

```bash
docker compose -f infra/docker-compose.monitoring.yaml up --build
```

This will start:

- The API service on port `8000`
- Prometheus on port `9090`
- Grafana on port `3000`

## Conclusion
This project demonstrates a complete end-to-end MLOps system. The emphasis is on production readiness, system design, and reproducibility.