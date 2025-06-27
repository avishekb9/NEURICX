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