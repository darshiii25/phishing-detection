"""Local HTTP adapter for the existing URL prediction helper.

Run from the phishing-detection project root with: python api_server.py
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from src.predict_url import predict_url


HOST = "127.0.0.1"
PORT = 8000
LOCAL_ORIGIN = re.compile(r"^http://(?:localhost|127\.0\.0\.1)(?::\d+)?$")


class PredictionHandler(BaseHTTPRequestHandler):
    server_version = "PhishGuardAPI/1.0"

    def _send_json(self, status: int, body: dict[str, object]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        origin = self.headers.get("Origin", "")
        if LOCAL_ORIGIN.fullmatch(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        origin = self.headers.get("Origin", "")
        if not LOCAL_ORIGIN.fullmatch(origin):
            self._send_json(403, {"error": "Origin not allowed"})
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/predict":
            self._send_json(404, {"error": "Endpoint not found"})
            return
        origin = self.headers.get("Origin", "")
        if origin and not LOCAL_ORIGIN.fullmatch(origin):
            self._send_json(403, {"error": "Origin not allowed"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 16_384:
                raise ValueError("Request body must be between 1 and 16384 bytes.")
            payload = json.loads(self.rfile.read(length))
            url = payload.get("url") if isinstance(payload, dict) else None
            if not isinstance(url, str) or not url.strip():
                raise ValueError("Provide a URL in the 'url' field.")
            url = url.strip()
            parts = urlsplit(url)
            if parts.scheme not in {"http", "https"} or not parts.hostname or "." not in parts.hostname:
                raise ValueError("Enter a complete http:// or https:// URL.")
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self._send_json(400, {"error": str(error) or "Invalid request."})
            return

        try:
            label, legitimate_probability, phishing_probability = predict_url(url)
        except Exception:
            self.log_error("Prediction failed")
            self._send_json(500, {"error": "The model could not analyze this URL. Check that model artifacts are available."})
            return

        self._send_json(200, {
            "prediction": label.lower(),
            "legitimate_probability": legitimate_probability,
            "phishing_probability": phishing_probability,
        })

    def log_message(self, format: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def log_error(self, message: str) -> None:
        print(f"[{self.log_date_time_string()}] {message}")


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), PredictionHandler)
    print(f"PhishGuard prediction API listening at http://{HOST}:{PORT}")
    print("POST /predict accepts JSON: {\"url\": \"https://example.com\"}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping PhishGuard prediction API.")
    finally:
        server.server_close()
