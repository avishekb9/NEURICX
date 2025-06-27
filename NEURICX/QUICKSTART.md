# 🚀 NEURICX Quick Start Guide

Get NEURICX running in minutes with multiple deployment options!

## 📋 Prerequisites Check

First, let's check what you have:

```bash
# Check your system
echo "OS: $(uname -s)"
echo "Architecture: $(uname -m)"

# Check available tools
command -v docker && echo "✅ Docker available" || echo "❌ Docker missing"
command -v R && echo "✅ R available" || echo "❌ R missing" 
command -v python3 && echo "✅ Python available" || echo "❌ Python missing"
```

## 🎯 Choose Your Setup Method

### Option 1: Quick Local Setup (No Docker Required) ⚡

**Best for:** Immediate testing, development, learning
**Time:** 5-10 minutes
**Requirements:** Python 3, R (optional)

```bash
# From NEURICX/deploy directory
./local-setup.sh

# After setup completes:
cd ../local-dev
./start-neuricx.sh
```

**Access:** http://localhost:3000/platform.html

### Option 2: Install Dependencies + Docker (Recommended) 🐳

**Best for:** Full functionality, production use
**Time:** 10-20 minutes
**Requirements:** Administrator access

```bash
# Step 1: Install dependencies
./install-dependencies.sh

# Step 2: Restart shell or run
newgrp docker  # Linux only

# Step 3: Deploy NEURICX
./deploy.sh
```

**Access:** http://localhost:3000/platform.html

### Option 3: Manual Docker Installation 🔧

If the automatic installer doesn't work:

#### Ubuntu/Debian:
```bash
# Update packages
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Restart shell, then:
./deploy.sh
```

#### macOS:
```bash
# Install Docker Desktop from:
# https://www.docker.com/products/docker-desktop

# Or using Homebrew:
brew install --cask docker

# Start Docker Desktop, then:
./deploy.sh
```

## 🎮 What You Can Do Right Away

Once NEURICX is running, try these features:

### 1. 🧠 Economic Simulation
- Click "Start Simulation" 
- Use default settings (1000 agents, 250 steps)
- Watch real-time progress
- View results in dashboard

### 2. 📊 Real-time Analytics  
- Click "Start Streaming"
- Monitor live market data
- Set custom alerts
- Interactive visualizations

### 3. ⚖️ Policy Analysis
- Click "Run Analysis"
- Try monetary policy (25 basis points)
- See network transmission effects
- Compare scenarios

### 4. ⚠️ Crisis Prediction
- Click "Predict Crisis"
- Use ensemble ML models
- Configure risk thresholds
- Monitor early warnings

### 5. ⚛️ Quantum Computing
- Click "Run Quantum"
- Start with simulator backend
- Try portfolio optimization
- Analyze quantum advantage

## 🔍 Verification Steps

Check everything is working:

```bash
# Test web interface
curl http://localhost:3000

# Test API server
curl http://localhost:8000/health

# Check all services (Docker setup)
docker-compose ps

# View logs if needed
docker-compose logs neuricx-api
```

## 🚨 Common Issues & Solutions

### Issue: "Docker not found"
```bash
# Solution: Install Docker first
./install-dependencies.sh
# Then restart your shell
```

### Issue: "Permission denied" (Docker)
```bash
# Solution: Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Issue: "Port already in use"
```bash
# Solution: Stop conflicting services
sudo lsof -i :3000  # Find what's using port 3000
sudo kill <PID>     # Stop the process
```

### Issue: "R packages won't install"
```bash
# Solution: Install system dependencies (Ubuntu)
sudo apt-get install libcurl4-openssl-dev libssl-dev libxml2-dev
```

### Issue: Web interface won't load
```bash
# Check if web server is running
curl http://localhost:3000

# For local setup, restart:
cd local-dev
./start-neuricx.sh
```

## 📊 System Requirements

### Minimum:
- **RAM:** 4GB (8GB recommended)
- **Storage:** 10GB free space
- **CPU:** 2+ cores
- **OS:** Linux, macOS, Windows 10+

### Recommended:
- **RAM:** 16GB+ for large simulations
- **Storage:** 50GB+ for data and logs  
- **CPU:** 8+ cores for parallel processing
- **Network:** Stable internet for real-time data

## 🎯 Next Steps

After getting NEURICX running:

1. **📖 Read the Guide:** Visit http://localhost:3000/launch-guide.html
2. **🎮 Try Examples:** Run example simulations with different parameters
3. **📊 Explore Dashboards:** Check out Grafana at http://localhost:3001
4. **🔧 Customize:** Modify configurations in `.env` file
5. **📈 Scale Up:** Try larger simulations and more complex analyses

## 🤝 Getting Help

### Built-in Help:
- **Launch Guide:** http://localhost:3000/launch-guide.html
- **System Status:** Click "System Status" in platform
- **Logs:** Click "View Logs" for troubleshooting

### External Resources:
- **Documentation:** Full API and usage documentation
- **GitHub Issues:** Report bugs and get community help
- **Examples:** Sample configurations and use cases

## 🔄 Development Workflow

For ongoing development:

```bash
# Start development environment
cd NEURICX/deploy
./local-setup.sh
cd ../local-dev
./start-neuricx.sh

# Make changes to web files
# Refresh browser to see changes

# For full stack development:
./deploy.sh  # Use Docker setup
# Edit files in NEURICX/ directory
# Restart containers as needed
```

## 🎉 Success!

If you see the NEURICX platform at http://localhost:3000/platform.html, you're ready to explore the future of economic modeling!

**Key URLs to bookmark:**
- 🏠 **Platform:** http://localhost:3000/platform.html
- 📊 **Dashboard:** http://localhost:3000/dashboard.html  
- 📖 **Guide:** http://localhost:3000/launch-guide.html
- 🔌 **API:** http://localhost:8000
- 📈 **Monitoring:** http://localhost:3001 (Docker setup)

---

**Need immediate help?** Run the appropriate setup script for your situation:
- ⚡ **Just want to try it:** `./local-setup.sh`
- 🐳 **Want full features:** `./install-dependencies.sh` then `./deploy.sh`
- 🔧 **Having issues:** Check the troubleshooting section above