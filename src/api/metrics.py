from prometheus_client import Counter, Histogram

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

NODE_PREDICTION_COUNTER = Counter(
    "predicted_node_total",
    "Count of predicted nodes",
    ["node"]
)
