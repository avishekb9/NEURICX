#!/bin/bash

# NEURICX Local Development Setup
# Quick setup without Docker for immediate testing

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "deploy.sh" ]; then
    print_error "Please run this script from the NEURICX/deploy directory"
    exit 1
fi

print_status "Setting up NEURICX for local development..."

# Create local directories
create_directories() {
    print_status "Creating local directories..."
    
    mkdir -p ../local-dev/{data,logs,config,web}
    
    print_success "Directories created"
}

# Setup R environment
setup_r_environment() {
    print_status "Setting up R environment..."
    
    # Check if R is installed
    if ! command -v R &> /dev/null; then
        print_warning "R is not installed. Installing R..."
        
        # Install R based on OS
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update
            sudo apt-get install -y r-base r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                brew install r
            else
                print_error "Please install R from https://cran.r-project.org/ or install Homebrew first"
                exit 1
            fi
        fi
    fi
    
    # Install required R packages
    print_status "Installing required R packages..."
    
    R --vanilla << 'EOF'
# Install required packages
required_packages <- c(
    "plumber", "jsonlite", "httr", "DBI", "RSQLite",
    "ggplot2", "plotly", "shiny", "shinydashboard", "DT",
    "igraph", "Matrix", "MASS", "dplyr", "tidyr",
    "devtools", "remotes"
)

install_if_missing <- function(packages) {
    for (pkg in packages) {
        if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
            cat("Installing", pkg, "...\n")
            install.packages(pkg, repos = "https://cran.rstudio.com/", quiet = TRUE)
        }
    }
}

install_if_missing(required_packages)
cat("R packages installed successfully!\n")
EOF

    print_success "R environment setup complete"
}

# Setup Python environment
setup_python_environment() {
    print_status "Setting up Python environment..."
    
    # Check if Python 3 is installed
    if ! command -v python3 &> /dev/null; then
        print_warning "Python 3 is not installed. Please install Python 3.8+ first"
        return 1
    fi
    
    # Install Python packages
    print_status "Installing Python packages..."
    pip3 install --user flask requests pandas numpy matplotlib seaborn scikit-learn
    
    print_success "Python environment setup complete"
}

# Setup web server
setup_web_server() {
    print_status "Setting up local web server..."
    
    # Copy web files to local development directory
    cp -r ../NEURICX_Web/* ../local-dev/web/
    
    # Create a simple Python web server script
    cat > ../local-dev/start-webserver.py << 'EOF'
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
EOF

    chmod +x ../local-dev/start-webserver.py
    
    print_success "Web server setup complete"
}

# Setup mock API server
setup_mock_api() {
    print_status "Setting up mock API server..."
    
    cat > ../local-dev/mock-api.py << 'EOF'
#!/usr/bin/env python3
from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import time
import threading

app = Flask(__name__)
CORS(app)

# Mock data storage
simulations = {}
jobs = {}

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0-dev",
        "services": {
            "database": True,
            "redis": False,
            "neuricx": True
        }
    })

@app.route('/economy/create', methods=['POST'])
def create_economy():
    data = request.json
    economy_id = f"economy_{int(time.time())}_{random.randint(1000, 9999)}"
    
    return jsonify({
        "economy_id": economy_id,
        "n_agents": data.get('n_agents', 1000),
        "symbols": data.get('symbols', ['AAPL', 'MSFT', 'GOOGL']),
        "network_density": data.get('network_density', 0.05),
        "created_at": time.time()
    })

@app.route('/simulation/run', methods=['POST'])
def run_simulation():
    data = request.json
    sim_id = f"simulation_{int(time.time())}_{random.randint(1000, 9999)}"
    
    # Store simulation job
    jobs[sim_id] = {
        "status": "running",
        "progress": 0,
        "economy_id": data.get('economy_id'),
        "n_steps": data.get('n_steps', 250),
        "start_time": time.time()
    }
    
    # Start background simulation
    def run_background_sim():
        for i in range(0, 101, 10):
            jobs[sim_id]["progress"] = i
            time.sleep(0.5)
        jobs[sim_id]["status"] = "completed"
    
    threading.Thread(target=run_background_sim).start()
    
    return jsonify({
        "simulation_id": sim_id,
        "economy_id": data.get('economy_id'),
        "status": "started",
        "n_steps": data.get('n_steps', 250)
    })

@app.route('/simulation/<sim_id>/results')
def get_simulation_results(sim_id):
    if sim_id not in jobs:
        return jsonify({"error": "Simulation not found"}), 404
    
    return jsonify({
        "simulation_id": sim_id,
        "status": jobs[sim_id]["status"],
        "progress": jobs[sim_id]["progress"],
        "summary": {
            "final_collective_intelligence": 1.15 + random.uniform(-0.1, 0.1),
            "average_wealth": 10000 + random.uniform(-1000, 1000),
            "wealth_inequality": 0.3 + random.uniform(-0.05, 0.05)
        }
    })

@app.route('/streaming/start', methods=['POST'])
def start_streaming():
    return jsonify({
        "stream_id": f"stream_{int(time.time())}",
        "status": "started",
        "message": "Real-time streaming initiated"
    })

if __name__ == '__main__':
    print("🔌 Starting NEURICX Mock API Server...")
    print("📡 API available at http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)
EOF

    chmod +x ../local-dev/mock-api.py
    
    # Install Flask if not present
    pip3 install --user flask flask-cors
    
    print_success "Mock API server setup complete"
}

# Create startup scripts
create_startup_scripts() {
    print_status "Creating startup scripts..."
    
    # Main startup script
    cat > ../local-dev/start-neuricx.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting NEURICX Local Development Environment..."

# Start mock API in background
echo "🔌 Starting Mock API Server..."
python3 mock-api.py &
API_PID=$!

# Wait a moment for API to start
sleep 2

# Start web server
echo "🌐 Starting Web Server..."
python3 start-webserver.py &
WEB_PID=$!

echo "✅ NEURICX Local Environment Started!"
echo ""
echo "🔗 Access Points:"
echo "   📱 Platform: http://localhost:3000/platform.html"
echo "   🎯 Dashboard: http://localhost:3000/dashboard.html"
echo "   📖 Guide: http://localhost:3000/launch-guide.html"
echo "   🔌 API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping NEURICX services..."
    kill $API_PID 2>/dev/null
    kill $WEB_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Set trap to cleanup on interrupt
trap cleanup INT

# Wait for interrupt
wait
EOF

    chmod +x ../local-dev/start-neuricx.sh
    
    # Quick test script
    cat > ../local-dev/test-setup.sh << 'EOF'
#!/bin/bash

echo "🧪 Testing NEURICX Local Setup..."

# Test R
if command -v R &> /dev/null; then
    echo "✅ R is available"
else
    echo "❌ R is not installed"
fi

# Test Python
if command -v python3 &> /dev/null; then
    echo "✅ Python 3 is available"
else
    echo "❌ Python 3 is not installed"
fi

# Test required Python packages
python3 -c "import flask, requests" 2>/dev/null && echo "✅ Python packages installed" || echo "❌ Python packages missing"

# Test web files
if [ -f "web/platform.html" ]; then
    echo "✅ Web files are in place"
else
    echo "❌ Web files missing"
fi

echo ""
echo "🚀 If all tests pass, run: ./start-neuricx.sh"
EOF

    chmod +x ../local-dev/test-setup.sh
    
    print_success "Startup scripts created"
}

# Main setup function
main() {
    echo "========================================="
    echo "  NEURICX Local Development Setup"
    echo "========================================="
    echo
    
    create_directories
    setup_r_environment
    setup_python_environment
    setup_web_server
    setup_mock_api
    create_startup_scripts
    
    echo
    echo "========================================="
    print_success "Local setup completed!"
    echo "========================================="
    echo
    print_status "Next steps:"
    echo "1. cd ../local-dev"
    echo "2. ./test-setup.sh  # Test the installation"
    echo "3. ./start-neuricx.sh  # Start NEURICX"
    echo ""
    print_status "The platform will be available at:"
    echo "   📱 http://localhost:3000/platform.html"
    echo ""
    print_warning "This is a development setup with mock data."
    print_warning "For full functionality, install Docker and use ./deploy.sh"
}

# Run main function
main "$@"