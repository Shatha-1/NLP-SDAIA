"""Lab 7: Python load-test equivalent of `hey -z 60s -c 16`.

The `hey` CLI (and Go, needed to build it) isn't available in this environment,
so this reproduces its exact load pattern -- fixed concurrency, fixed wall-clock
duration, same endpoint/payload -- using a thread pool, and reports the same
p50/p99/error-count style summary. See load_test_config.yaml for parameters.
"""
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml

CONFIG_PATH = Path("data/serving/load_test_config.yaml")
PAYLOAD = {"text": "الخدمة ممتازة ولكن التأخير طويل"}


def _worker(endpoint: str, stop_at: float, latencies: list, errors: list, lock: threading.Lock):
    session = requests.Session()
    while time.monotonic() < stop_at:
        start = time.perf_counter()
        try:
            resp = session.post(endpoint, json=PAYLOAD, timeout=10)
            elapsed_ms = (time.perf_counter() - start) * 1000
            with lock:
                latencies.append(elapsed_ms)
                if resp.status_code != 200:
                    errors.append(resp.status_code)
        except requests.RequestException as exc:
            with lock:
                errors.append(str(exc))


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    endpoint = config["endpoint"]
    concurrency = config["concurrency"]
    duration = config["duration_seconds"]
    target_p99_ms = config["target_p99_ms"]

    print(f"Load testing {endpoint} -- {concurrency} concurrent workers, {duration}s")

    latencies: list = []
    errors: list = []
    lock = threading.Lock()
    stop_at = time.monotonic() + duration

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_worker, endpoint, stop_at, latencies, errors, lock) for _ in range(concurrency)
        ]
        for f in futures:
            f.result()

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.50)] if n else float("nan")
    p99 = latencies[min(int(n * 0.99), n - 1)] if n else float("nan")
    mean = statistics.mean(latencies) if n else float("nan")

    print(f"\nTotal requests: {n}")
    print(f"Errors: {len(errors)}")
    print(f"Mean latency: {mean:.2f} ms")
    print(f"p50 latency:  {p50:.2f} ms")
    print(f"p99 latency:  {p99:.2f} ms (target <= {target_p99_ms} ms)")
    print(f"Target met: {p99 <= target_p99_ms and len(errors) == 0}")


if __name__ == "__main__":
    main()
