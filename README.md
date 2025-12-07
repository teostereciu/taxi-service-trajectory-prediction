# taxi-service-trajectory-prediction

FasAPI example

```
curl -X POST http://localhost:8000/predict_next_node \
  -H "Content-Type: application/json" \
  -d '{
    "prev_node": "ez3fh5",
    "curr_node": "ez3fhh",
    "hour": 17,
    "day_of_week": 3,
    "call_type": "B"
  }'
```