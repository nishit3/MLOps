"""Container HEALTHCHECK: exit 0 if the API answers /health, else 1.

Uses only the standard library because python:slim images don't ship curl.
"""
import os
import sys
import urllib.request

port = os.environ.get("PORT", "8000")
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
        sys.exit(0 if resp.status == 200 else 1)
except Exception:
    sys.exit(1)
