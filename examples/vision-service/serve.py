from http.server import BaseHTTPRequestHandler, HTTPServer
import json


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/healthz'):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')
            return

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'prediction': 'cat', 'confidence': 0.98}).encode('utf-8'))


if __name__ == '__main__':
    HTTPServer(('0.0.0.0', 8080), Handler).serve_forever()
