import time
import random
import requests
import math
from datetime import datetime

API_URL = "http://localhost:8000/predict_next_node"

CORE_NODES = [
    "ez3fhj", "ez3fhk", "ez3fhh", "ez3f5u", "ez3fhn"
]

RARE_NODES = [
    "ez3fht", "ez3fhp", "ez3fhm", "ez3f5s"
]

INVALID_NODE = "invalid"

CALL_TYPES = ["A", "B", "C"]


def current_load_factor(t: float) -> float:
    """
    Sinusoidal load pattern:
    - simulates natural traffic waves
    """
    return 0.5 + 0.5 * math.sin(t / 30.0)

def pick_node(drift_factor: float):
    """
    Gradually bias toward rare nodes as drift increases
    """
    if random.random() < 0.05 + drift_factor:
        return random.choice(RARE_NODES)
    return random.choice(CORE_NODES)

def maybe_invalid():
    """
    Inject invalid inputs occasionally
    """
    return INVALID_NODE if random.random() < 0.03 else None


def make_request(drift_factor: float):
    now = datetime.utcnow()
    hour = now.hour
    dow = now.weekday()

    if 9 <= hour <= 17:
        call_type = random.choices(CALL_TYPES, weights=[0.6, 0.3, 0.1])[0]
    else:
        call_type = random.choices(CALL_TYPES, weights=[0.2, 0.3, 0.5])[0]

    prev_node = pick_node(drift_factor)
    curr_node = pick_node(drift_factor)

    if random.random() < 0.05:
        prev_node = maybe_invalid() or prev_node

    payload = {
        "prev_node": prev_node,
        "curr_node": curr_node,
        "call_type": call_type,
        "hour": hour,
        "day_of_week": dow,
    }

    try:
        r = requests.post(API_URL, json=payload, timeout=2)
        print(f"{r.status_code} → {r.json()}")
    except Exception as e:
        print("REQUEST ERROR:", e)


if __name__ == "__main__":
    print("🚕 Sending requests… Ctrl+C to stop")

    start = time.time()

    while True:
        elapsed = time.time() - start

        drift_factor = min(0.3, elapsed / 600)

        load = current_load_factor(elapsed)

        make_request(drift_factor)

        base_sleep = random.uniform(0.2, 1.0)
        sleep_time = max(0.1, base_sleep * (1.5 - load))

        time.sleep(sleep_time)
