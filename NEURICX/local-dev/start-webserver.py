#!/usr/bin/env python3
import http.server
import socketserver
import os
import webbrowser
from threading import Timer

PORT = 3000
DIRECTORY = "web"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def open_browser():
    webbrowser.open(f'http://localhost:{PORT}/platform.html')

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🌐 Web server running at http://localhost:{PORT}")
        print(f"📱 Opening NEURICX Platform at http://localhost:{PORT}/platform.html")
        
        # Open browser after 2 seconds
        Timer(2.0, open_browser).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Web server stopped")
            httpd.shutdown()