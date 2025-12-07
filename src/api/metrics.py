from prometheus_client import Counter, Histogram, Gauge, Info

# Requests

REQUEST_COUNTER = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

ERROR_COUNTER = Counter(
    "prediction_errors_total",
    "Number of prediction errors"
)

LATENCY_HISTOGRAM = Histogram(
    "prediction_latency_seconds",
    "Latency of prediction requests"
)

# Model behaviour

NODE_PREDICTION_COUNTER = Counter(
    "predicted_node_total",
    "Count of predicted nodes",
    ["node"]
)


# Model metadata

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Model accuracy on test set"
)

MODEL_TOP3_ACCURACY = Gauge(
    "model_top3_accuracy",
    "Model top-3 accuracy on test set"
)

MODEL_INFO = Info(
    "model_info",
    "Model metadata"
)


# Model uncertainty

MODEL_CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "Max predicted probability per request"
)

MODEL_ENTROPY = Histogram(
    "model_prediction_entropy",
    "Entropy of predicted probability distribution"
)