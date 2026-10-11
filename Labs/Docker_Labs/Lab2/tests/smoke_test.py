"""End-to-end smoke test, run as its own container by docker compose.

It talks to the serving container over the compose network (http://serving:PORT),
so a pass proves: images built, training passed its quality gate, the model
reached serving through the shared volume, and the API answers correctly.
Standard library only, so it runs on a plain python:3.10-slim image.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("BASE_URL", "http://serving:8000").rstrip("/")

# Typical, well-separated examples of each species
SAMPLES = [
    ("Adelie", {"bill_length_mm": 39.1, "bill_depth_mm": 18.7, "flipper_length_mm": 181,
                "body_mass_g": 3750, "island": "Torgersen", "sex": "male"}),
    ("Chinstrap", {"bill_length_mm": 46.5, "bill_depth_mm": 17.9, "flipper_length_mm": 192,
                   "body_mass_g": 3500, "island": "Dream", "sex": "female"}),
    ("Gentoo", {"bill_length_mm": 47.5, "bill_depth_mm": 15.0, "flipper_length_mm": 217,
                "body_mass_g": 5200, "island": "Biscoe", "sex": "male"}),
]


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def main():
    failures = 0

    # serving reports healthy before this container starts, but retry briefly anyway
    for _ in range(10):
        try:
            status, body = request("/health")
            if status == 200:
                break
        except OSError:
            pass
        time.sleep(2)
    else:
        print("[smoke] FAIL: /health never answered")
        return 1
    print(f"[smoke] /health ok, model_version={body.get('model_version')}")

    status, meta = request("/metadata")
    print(f"[smoke] /metadata -> {status}, test_accuracy={meta.get('test_accuracy')}, "
          f"selected={meta.get('selected_model', {}).get('hidden_units')}")
    failures += status != 200

    for expected, sample in SAMPLES:
        status, body = request("/predict", sample)
        ok = status == 200 and body.get("predicted_class") == expected
        failures += not ok
        print(f"[smoke] {'PASS' if ok else 'FAIL'}: expected {expected}, got "
              f"{body.get('predicted_class')} (confidence {body.get('confidence')})")

    status, body = request("/predict", {"bill_length_mm": "abc"})
    ok = status == 400
    failures += not ok
    print(f"[smoke] {'PASS' if ok else 'FAIL'}: invalid input rejected with {status}: {body.get('error')}")

    print(f"[smoke] {'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
