# utils.py
from datetime import datetime

def log_event(action, path="activity_log.txt"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{now}] {action}"
    print(entry)
    with open(path, "a") as f:
        f.write(entry + "\n")
