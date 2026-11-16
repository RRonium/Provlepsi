"""
Optional local engine for Provlepsi.

By default the app trains the Python models inside the browser with Pyodide.
For big datasets you can run them with your own desktop Python instead:

    pip install -r python/requirements.txt
    python python/local_server.py            # listens on http://127.0.0.1:8765

then pick "Local Python" in the app's engine switch. The server only listens on
127.0.0.1, only runs the model files in python/models, and only answers the
origins listed in PROVLEPSI_ORIGINS, so your data still never leaves your machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import provlepsi_runner as runner  # noqa: E402

MODELS_DIR = HERE / "models"
DATA_DIR = Path(tempfile.mkdtemp(prefix="provlepsi-"))
DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = {
    o.strip() for o in os.environ.get("PROVLEPSI_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()
}
MAX_BODY = 512 * 1024 * 1024


def _model_files() -> dict[str, Path]:
    return {p.name: p for p in MODELS_DIR.glob("*_model.py")}


class Handler(BaseHTTPRequestHandler):
    server_version = "ProvlepsiLocal/1.0"

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin in ALLOWED_ORIGINS

    def _send(self, status: int, payload: str) -> None:
        body = payload.encode()
        self.send_response(status)
        origin = self.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # CORS preflight
        if not self._origin_ok():
            self.send_response(403)
            self.end_headers()
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin", ""))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        # Lets Chrome reach this localhost server from a public https page (e.g. GitHub Pages).
        if self.headers.get("Access-Control-Request-Private-Network") == "true":
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._origin_ok():
            return self._send(403, json.dumps({"ok": False, "error": "Origin not allowed"}))
        if self.path.rstrip("/") == "/health":
            import pandas, sklearn  # noqa: E401

            versions = {"python": sys.version.split()[0], "pandas": pandas.__version__, "sklearn": sklearn.__version__}
            return self._send(200, json.dumps({"ok": True, "data": {"versions": versions, "models": sorted(_model_files())}}))
        self._send(404, json.dumps({"ok": False, "error": "Not found"}))

    def do_POST(self) -> None:
        # Requiring JSON forces browsers to send a CORS preflight, so other sites can't post here.
        if not self._origin_ok() or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self._send(403, json.dumps({"ok": False, "error": "Forbidden"}))
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            return self._send(413, json.dumps({"ok": False, "error": "Dataset too large"}))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, json.dumps({"ok": False, "error": "Bad JSON"}))

        if self.path == "/run":
            files = _model_files()
            model = files.get(str(req.get("model_file")))
            if model is None:
                return self._send(400, json.dumps({"ok": False, "error": "Unknown model file"}))
            csv_path = DATA_DIR / "dataset.csv"
            csv_path.write_text(req.get("csv", ""))
            config = dict(req.get("config") or {})
            config["csv_path"] = str(csv_path)
            out = runner.run_model_json(model.read_text(), json.dumps(config), model.name)
            return self._send(200, out)
        if self.path == "/predict":
            out = runner.predict_json(str(req.get("session_id")), json.dumps(req.get("entry") or {}))
            return self._send(200, out)
        self._send(404, json.dumps({"ok": False, "error": "Not found"}))

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[provlepsi] " + (fmt % args) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Provlepsi local engine on http://127.0.0.1:{args.port} (origins: {', '.join(sorted(ALLOWED_ORIGINS))})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
