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