# NEURICX Web Interface

Complete web-based platform for launching and managing all NEURICX capabilities including infrastructure deployment, economic simulations, real-time analytics, and advanced modeling tools.

## 🚀 Quick Launch

1. **Start the Web Platform**
   ```bash
   # Open in your browser
   http://localhost:3000/platform.html
   ```

2. **Launch Infrastructure**
   - Click "Deploy Infrastructure" 
   - Select deployment environment (development/production)
   - Choose cloud provider or Docker
   - Click "Deploy" and monitor progress

3. **Start Economic Simulation**
   - Configure agents, symbols, and network parameters
   - Launch simulation and view real-time progress
   - Access results through interactive dashboards

## 📁 File Structure

```
NEURICX_Web/
├── index.html              # Main landing page
├── platform.html           # Primary launch interface ⭐
├── dashboard.html           # Analytics dashboard
├── launch-guide.html        # Complete setup guide
├── startup.js              # Platform initialization
├── modals.js               # Modal dialogs and forms
├── additional-styles.css    # Extended styling
└── README.md               # This file
```

## 🎯 Key Features

### 1. Infrastructure Management
- **One-click deployment** of entire NEURICX stack
- **Multi-cloud support** (AWS, GCP, Azure, local Docker)
- **Real-time status monitoring** of all services
- **Automatic health checks** and service recovery

**How to Use:**
1. Open `platform.html` 
2. Click "Deploy Infrastructure" in the Infrastructure Management card
3. Configure your deployment settings:
   - Environment: development/staging/production
   - Cloud Provider: docker/aws/gcp/azure
   - Domain Name: (optional for custom domains)
   - Enable SSL/TLS and monitoring options
4. Click "Deploy Infrastructure"
5. Monitor deployment progress in real-time
6. Access services once deployment completes

### 2. Economic Simulation
- **Multi-agent modeling** with 6 heterogeneous agent types
- **Network evolution** across production, consumption, and information layers
- **Real-time visualization** of simulation progress
- **Performance benchmarking** against traditional models

**How to Use:**
1. Click "Start Simulation" in the Economic Simulation card
2. Configure simulation parameters:
   - Number of Agents: 100-10,000 (default: 1000)
   - Simulation Steps: 50-1000 (default: 250)
   - Stock Symbols: comma-separated list
   - Network Density: 0.01-0.2 (default: 0.05)
3. Click "Start Simulation"
4. Monitor progress and view results in dashboard

### 3. Real-Time Analytics
- **Live market data** streaming from multiple sources
- **Interactive dashboards** with D3.js visualizations
- **Custom alerts** and notifications
- **WebSocket integration** for real-time updates

**How to Use:**
1. Click "Start Streaming" in the Real-time Analytics card
2. Configure data sources and symbols
3. Set update frequency and buffer size
4. Click "Start Streaming"
5. View live data in the dashboard

### 4. Policy Analysis
- **Comprehensive policy modeling** (monetary, fiscal, regulatory, technology)
- **Intervention simulation** with network transmission analysis
- **Counterfactual analysis** comparing baseline vs policy scenarios
- **Impact assessment** across different agent types

**How to Use:**
1. Click "Run Analysis" in the Policy Analysis card
2. Select policy type and configure parameters
3. Set intervention timing and duration
4. Click "Run Policy Analysis"
5. Review results and transmission effects

### 5. Crisis Prediction
- **Ensemble ML models** for crisis prediction
- **Early warning systems** with configurable thresholds
- **Systemic risk assessment** and monitoring
- **Real-time alerts** for potential crises

**How to Use:**
1. Click "Predict Crisis" in the Crisis Prediction card
2. Configure prediction horizon and models
3. Set risk thresholds for different indicators
4. Enable real-time monitoring if desired
5. Click "Predict Crisis"
6. Monitor risk levels and alerts

### 6. Quantum Computing
- **Quantum optimization** algorithms (QAOA, VQE, quantum annealing)
- **Multiple backend support** (IBM, Rigetti, IonQ, simulators)
- **Portfolio optimization** and network analysis
- **Quantum machine learning** capabilities

**How to Use:**
1. Click "Run Quantum" in the Quantum Computing card
2. Select quantum backend and number of qubits
3. Choose algorithm (portfolio, network, ML)
4. Configure algorithm-specific parameters
5. Click "Run Quantum Algorithm"
6. View quantum advantage analysis

### 7. Machine Learning
- **Ensemble methods** with multiple base models
- **Automated hyperparameter optimization**
- **Cross-validation** and performance evaluation
- **Model comparison** and selection

**How to Use:**
1. Click "Train Models" in the Machine Learning card
2. Select base models and ensemble method
3. Choose target variable and training configuration
4. Enable automatic hyperparameter optimization
5. Click "Train ML Ensemble"
6. View model performance and results

### 8. Monitoring & Observability
- **System metrics** with Prometheus and Grafana
- **Real-time dashboards** for performance monitoring
- **Log aggregation** and analysis
- **Custom alerts** and notifications

**How to Use:**
1. Click "Open Grafana" to access monitoring dashboards
2. Click "View Logs" to see system logs
3. Configure alerts and thresholds
4. Monitor system health and performance

## 🔧 Configuration

### Environment Variables
Set these in your deployment configuration:

```bash
# Core Configuration
NEURICX_ENV=development          # or staging/production
DOMAIN=localhost                 # your domain name
SSL_ENABLED=false               # enable SSL/TLS
MONITORING_ENABLED=true         # enable monitoring stack

# API Keys (optional)
ALPHA_VANTAGE_API_KEY=your_key
POLYGON_API_KEY=your_key
QISKIT_IBM_TOKEN=your_token
IONQ_API_TOKEN=your_token

# Database
POSTGRES_PASSWORD=secure_password
REDIS_PASSWORD=secure_password
```

### Service Endpoints
After deployment, services are available at:

- **Main Platform**: http://localhost:3000/platform.html
- **API Server**: http://localhost:8000
- **Real-time Streaming**: ws://localhost:8001
- **Quantum Service**: http://localhost:8002
- **Grafana Monitoring**: http://localhost:3001
- **Prometheus Metrics**: http://localhost:9090
- **Jupyter Lab**: http://localhost:8888

## 🎨 User Interface Guide

### Main Platform Interface (`platform.html`)

#### Header Section
- **NEURICX Platform Logo**: Returns to main interface
- **System Status Indicators**: Shows API, Streaming, and Quantum service status
- **Real-time Statistics**: Active simulations, total agents, quantum jobs, uptime

#### Platform Overview
- **Quick Statistics**: System overview and current activity
- **Launch Buttons**: Direct access to all capabilities

#### Capability Cards
Each capability has its own card with:
- **Description**: What the capability does
- **Feature List**: Key features and benefits
- **Action Buttons**: Primary action (e.g., "Start Simulation") and secondary action (e.g., "View Dashboard")

#### Modal Dialogs
Each capability opens a configuration modal with:
- **Parameter Settings**: Configurable options for the capability
- **Progress Tracking**: Real-time progress updates during execution
- **Status Messages**: Success, error, and warning notifications
- **Result Links**: Direct links to view results in dashboards

### Dashboard Integration
- **Seamless Navigation**: One-click access to relevant dashboards
- **Real-time Updates**: Live data and status updates
- **Interactive Visualizations**: D3.js-powered charts and graphs
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## 🚨 Troubleshooting

### Common Issues

1. **Services Not Starting**
   ```bash
   # Check Docker is running
   docker --version
   
   # Check port availability
   netstat -tulpn | grep :8000
   
   # View service logs
   docker-compose logs neuricx-api
   ```

2. **Cannot Access Web Interface**
   - Ensure ports 3000, 8000, 8001 are not blocked
   - Check firewall settings
   - Verify all services are running: `docker-compose ps`

3. **API Calls Failing**
   - Check API service status in the header
   - Verify backend services are running
   - Check browser console for detailed error messages

4. **WebSocket Connection Issues**
   - Ensure streaming service is running
   - Check firewall allows WebSocket connections
   - Try refreshing the page

### Getting Help

1. **Built-in Help**: Click help icons (ℹ️) for contextual information
2. **Launch Guide**: Visit `launch-guide.html` for complete setup instructions
3. **System Status**: Use the "System Status" button to check service health
4. **Logs**: Access "View Logs" to see detailed system information

## 🔗 Integration with NEURICX Components

The web interface integrates with all NEURICX R package components:

- **R API Server** (`neuricx_*.R` files): Backend computation engine
- **Docker Infrastructure** (`deploy/` directory): Containerized deployment
- **Monitoring Stack** (Prometheus/Grafana): System observability
- **Database Layer** (PostgreSQL/Redis): Data persistence and caching

## 📊 Performance Optimization

### For Large Simulations
- Use fewer agents initially (100-500) to test
- Increase gradually based on system performance
- Monitor memory and CPU usage
- Consider distributed deployment for >5000 agents

### For Real-time Streaming
- Start with longer update frequencies (60+ seconds)
- Reduce frequency as needed for your use case
- Monitor network bandwidth usage
- Use data source rate limits appropriately

### For Quantum Computing
- Start with simulator backend for testing
- Use fewer qubits for initial experiments
- Monitor quantum service resource usage
- Consider queue times for real quantum hardware

## 🎯 Next Steps

1. **Explore the Platform**: Start with `platform.html` and try each capability
2. **Run a Simple Simulation**: Use default parameters for your first simulation
3. **Monitor Performance**: Check system metrics in Grafana
4. **Customize Configuration**: Adjust parameters based on your needs
5. **Scale Up**: Increase complexity gradually as you become familiar

## 🤝 Contributing

To extend the web interface:

1. **Add New Capabilities**: Create modal dialogs in `modals.js`
2. **Enhance Visualizations**: Extend `dashboard.html` with new charts
3. **Improve Styling**: Add custom styles to `additional-styles.css`
4. **Add Features**: Extend `startup.js` with new functionality

---

**Ready to Launch NEURICX? Start with [platform.html](platform.html) and explore the future of economic modeling!** 🚀