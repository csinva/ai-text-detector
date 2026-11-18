#!/usr/bin/env python3
"""Static server for local tests of the app, with the COOP/COEP headers that coi-serviceworker emulates on GitHub Pages.  python app/tools/serve.py [port] [--no-isolation]"""
import http.server, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); ISO = "--no-isolation" not in sys.argv


class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript", ".wasm": "application/wasm", ".js": "text/javascript"}

    def __init__(self, *a, **k): super().__init__(*a, directory=ROOT, **k)

    def end_headers(self):
        if ISO: self.send_header("Cross-Origin-Opener-Policy", "same-origin"); self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache"); super().end_headers()

    def log_message(self, *a): pass


if __name__ == "__main__":
    port = int([a for a in sys.argv[1:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[1:]) else 8765
    http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
