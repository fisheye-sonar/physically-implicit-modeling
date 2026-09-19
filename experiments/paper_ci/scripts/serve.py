#!/usr/bin/env python3
"""serve.py — the dashboard's static file server (127.0.0.1:<port>, no caching, no listing).
``tailscale serve`` publishes it on the tailnet over HTTPS; the page fetches state.json beside it.

    python experiments/paper_ci/scripts/serve.py 8765     (cwd = the dashboard directory)
"""
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class H(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404)
        return None

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
