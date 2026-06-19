"""Default, dependency-free transport: a stdlib http.server wrapping PredictionService.

This guarantees the project runs and is testable with ONLY the baseline libs (no
FastAPI/uvicorn needed). The optional FastAPI app in `app.py` shares the exact same
PredictionService, so the security behaviour is identical.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .service import PredictionService

API_KEY_HEADER = "x-api-key"
MAX_BODY_BYTES = 256 * 1024  # reject oversized bodies before parsing (DoS guard)


def make_handler(service: PredictionService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # silence default stderr spam
            pass

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            resp = service.handle(self.headers.get(API_KEY_HEADER), self.path, None)
            self._send(resp.status, resp.body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self._send(413, {"error": "payload too large"})
                return
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid JSON"})
                return
            api_key = self.headers.get(API_KEY_HEADER)
            resp = service.handle(api_key, self.path, body)
            self._send(resp.status, resp.body)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8077, **build_kwargs) -> None:
    service = PredictionService.build(**build_kwargs)
    server = ThreadingHTTPServer((host, port), make_handler(service))
    print(f"serving on http://{host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
