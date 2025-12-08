import time
import random
import requests

API_URL = "http://localhost:8000/predict_next_node"

NODES = [
    "ez3fhj",
    "ez3fhk",
    "ez3fhh",
    "ez3f5u",
    "invalid"
]

def make_request():
    payload = {
        "prev_node": random.choice(NODES),
        "curr_node": random.choice(NODES),
        "call_type": random.choice(["A", "B", "C"]),
        "hour": random.randint(0, 23),
        "day_of_week": random.randint(0, 6),
    }

    try:
        r = requests.post(API_URL, json=payload, timeout=2)
        print(r.json())
    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    print("Sending requests… ctrl+c to stop")
    while True:
        make_request()
        time.sleep(random.uniform(0.3, 1.5))
