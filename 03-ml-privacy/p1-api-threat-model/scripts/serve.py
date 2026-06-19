#!/usr/bin/env python3
"""Run the hardened model-serving endpoint.

Default transport is the dependency-free stdlib HTTP server. Pass --fastapi to use
the optional FastAPI/uvicorn transport (requires `pip install fastapi uvicorn`).

Example client call (default port 8077):
    curl -s -X POST http://127.0.0.1:8077/predict \\
         -H 'x-api-key: demo-secret-key' -H 'content-type: application/json' \\
         -d '{"instances": [[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]]}'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the hardened /predict endpoint.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--fastapi", action="store_true", help="use optional FastAPI transport")
    ap.add_argument("--capacity", type=int, default=5, help="token-bucket capacity")
    ap.add_argument("--refill", type=float, default=1.0, help="tokens refilled per second")
    args = ap.parse_args()

    if args.fastapi:
        try:
            import uvicorn

            from api_threat_model import create_app
        except ImportError:
            print(
                "FastAPI/uvicorn not installed. Run without --fastapi (stdlib transport) "
                "or `pip install fastapi uvicorn`.",
                file=sys.stderr,
            )
            sys.exit(1)
        app = create_app(capacity=args.capacity, refill_per_sec=args.refill)
        print(f"serving (FastAPI) on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    else:
        from api_threat_model import serve

        serve(
            host=args.host,
            port=args.port,
            capacity=args.capacity,
            refill_per_sec=args.refill,
        )


if __name__ == "__main__":
    main()
