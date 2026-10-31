"""Integration adapter for the PhishGuard frontend and the phishing-detection backend.

This file is part of a SEPARATE integration layer. It does NOT modify the
frontend or backend codebases. It imports the backend's existing
``predict_url`` function (read-only use of backend code) and exposes it over a
minimal HTTP endpoint that matches the frontend's existing contract.

The frontend (src/api.ts) sends ``POST {baseUrl}/predict`` with body
``{"url": "<string>"}`` and expects a JSON response shaped like:

    {"prediction": "phishing" | "legitimate",
     "legitimate_probability": <float 0..1>,
     "phishing_probability": <float 0..1>}

The backend's ``predict_url()`` returns ``(label, legit_prob, phish_prob)``
where ``label`` is the capitalized string ``"Legitimate"`` / ``"Phishing"`` and
the two probabilities are floats in [0, 1]. This adapter simply lowercases the
label and forwards the probabilities verbatim. No ``reasons`` field is emitted
because the backend does not produce one; the frontend already handles that
absence gracefully.

Run this with the SAME Python interpreter / virtual environment as the backend
(it needs tensorflow, numpy, and the saved model artifacts under
``models/`` and ``outputs/``).

    python server.py

Environment variables:
    BACKEND_DIR       Absolute path to the backend project root (the folder
                      containing ``config.py`` and ``src/``). Defaults to the
                      known location used during integration.
    INTEGRATION_HOST  Host to bind. Defaults to ``127.0.0.1``.
    INTEGRATION_PORT  Port to listen on. Defaults to ``8000``, which matches
                      the frontend's default ``VITE_API_BASE_URL`` so no frontend
                      configuration change is required.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Backend root resolved relative to this file so the adapter is location-independent
# and does not depend on a previously-hardcoded absolute path.
# server.py lives in <backend>/src/server.py, so parents[1] is the backend root.
DEFAULT_BACKEND_DIR = str(Path(__file__).resolve().parents[1])

BACKEND_DIR = Path(os.environ.get("BACKEND_DIR", DEFAULT_BACKEND_DIR)).resolve()
HOST = os.environ.get("INTEGRATION_HOST", "127.0.0.1")
PORT = int(os.environ.get("INTEGRATION_PORT", "8000"))

# Lazily imported backend function (importing it pulls in tensorflow).
_predict_url = None
_import_lock = threading.Lock()
# Serializes actual predictions to keep repeated Keras model loads safe.
_predict_lock = threading.Lock()


def get_predict_url():
    """Import and cache the backend's ``predict_url`` function on first use."""
    global _predict_url
    if _predict_url is None:
        with _import_lock:
            if _predict_url is None:
                backend_root = str(BACKEND_DIR)
                if backend_root not in sys.path:
                    sys.path.insert(0, backend_root)
                # Read-only import of the backend package; no backend file changes.
                from src.predict_url import predict_url  # type: ignore

                _predict_url = predict_url
    return _predict_url


class Handler(BaseHTTPRequestHandler):
    server_version = "PhishGuardIntegration/1.0"
    protocol_version = "HTTP/1.1"

    # -- helpers ----------------------------------------------------------
    def _cors(self) -> None:
        # The browser calls port 8000 from the Vite origin (e.g. 5173), so the
        # response must be CORS-permissive. The backend has no CORS settings to
        # change because it has no web server; this header lives only here.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    # -- HTTP methods -----------------------------------------------------
    def do_OPTIONS(self) -> None:  # CORS preflight
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "Not found. Use POST /predict."})

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/predict":
            self._json(404, {"error": "Not found. Use POST /predict."})
            return

        # Read request body using the declared Content-Length.
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""

        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(400, {"error": "Request body must be valid JSON."})
            return

        if not isinstance(data, dict):
            self._json(400, {"error": "Request body must be a JSON object."})
            return

        url = data.get("url")
        if not isinstance(url, str) or not url.strip():
            self._json(400, {"error": "Field 'url' is required and must be a non-empty string."})
            return

        # Call the real backend prediction function. Never fabricate results.
        try:
            with _predict_lock:
                label, legit_prob, phish_prob = get_predict_url()(url.strip())
        except Exception as exc:  # noqa: BLE001 - surface backend failures verbatim
            self._json(500, {"error": f"Prediction failed: {exc}"})
            return

        prediction = str(label).strip().lower()
        if prediction not in ("phishing", "legitimate"):
            self._json(500, {"error": f"Unexpected label from backend: {label!r}"})
            return

        self._json(200, {
            "prediction": prediction,
            "legitimate_probability": float(legit_prob),
            "phishing_probability": float(phish_prob),
        })

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def _preflight_checks() -> None:
    """Warn (but do not abort) if the backend looks incomplete at startup."""
    if not BACKEND_DIR.is_dir():
        sys.stderr.write(
            "WARNING: BACKEND_DIR does not exist: %s\n"
            "Set the BACKEND_DIR environment variable to the backend project root.\n"
            % BACKEND_DIR
        )
        return
    missing = [
        name for name in ("config.py", "src", "models", "outputs")
        if not (BACKEND_DIR / name).exists()
    ]
    if missing:
        sys.stderr.write(
            "WARNING: BACKEND_DIR looks incomplete; missing: %s\n" % ", ".join(missing)
        )


def main() -> None:
    _preflight_checks()
    print("PhishGuard integration adapter")
    print("  Backend dir : %s" % BACKEND_DIR)
    print("  Listening on: http://%s:%s" % (HOST, PORT))
    print("  Endpoint    : POST http://%s:%s/predict  (body: {\"url\": \"...\"})" % (HOST, PORT))
    print("  Health check: GET  http://%s:%s/health" % (HOST, PORT))
    print("(Ctrl+C to stop)")
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
