// Additional Modal Dialogs for NEURICX Platform
// This file contains the remaining modal implementations

// Check if we're in demo mode
const isDemoMode = window.location.hostname.includes('github.io');
const launcher = isDemoMode ? 
    (() => window.neuricxDemo) : 
    (() => window.neuricxLauncher);

// Policy Analysis Modal
function showPolicyModal() {
    const modal = createModal('policyModal', 'Policy Analysis Configuration', `
        <form id="policyForm">
            <div class="form-group">
                <label class="form-label">Policy Type</label>
                <select class="form-control" id="policyType">
                    <option value="monetary">Monetary Policy</option>
                    <option value="fiscal">Fiscal Policy</option>
                    <option value="regulatory">Regulatory Policy</option>
                    <option value="technology">Technology Policy</option>
                </select>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">Policy Intensity</label>
                    <input type="range" class="form-control" id="policyIntensity" min="0.001" max="0.1" step="0.001" value="0.02">
                    <small>Current: <span id="intensityValue">0.02</span></small>
                </div>
                <div class="form-group">
                    <label class="form-label">Intervention Timing (Step)</label>
                    <input type="number" class="form-control" id="interventionTiming" value="50" min="1" max="200">
                </div>
            </div>
            <div class="form-group">
                <label class="form-label">Duration (Steps)</label>
                <input type="number" class="form-control" id="policyDuration" value="25" min="1" max="100">
            </div>
            <div class="form-group" id="monetaryOptions" style="display: block;">
                <label class="form-label">Interest Rate Change (Basis Points)</label>
                <input type="number" class="form-control" id="interestRateChange" value="25" min="-500" max="500">
            </div>
            <div class="form-group" id="fiscalOptions" style="display: none;">
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Spending Change (% GDP)</label>
                        <input type="number" class="form-control" id="spendingChange" value="0.02" min="-0.1" max="0.1" step="0.001">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Tax Change (Percentage Points)</label>
                        <input type="number" class="form-control" id="taxChange" value="0" min="-0.05" max="0.05" step="0.001">
                    </div>
                </div>
            </div>
            <div class="form-group" id="regulatoryOptions" style="display: none;">
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Regulation Strength</label>
                        <input type="range" class="form-control" id="regulationStrength" min="0.001" max="0.1" step="0.001" value="0.03">
                        <small>Current: <span id="regStrengthValue">0.03</span></small>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Regulation Scope</label>
                        <select class="form-control" id="regulationScope">
                            <option value="network">Network Structure</option>
                            <option value="agents">Agent Behavior</option>
                            <option value="markets">Market Operations</option>
                        </select>
                    </div>
                </div>
            </div>
            <div class="form-group" id="technologyOptions" style="display: none;">
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Innovation Boost</label>
                        <input type="range" class="form-control" id="innovationBoost" min="0.001" max="0.2" step="0.001" value="0.05">
                        <small>Current: <span id="innovationValue">0.05</span></small>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Diffusion Rate</label>
                        <input type="range" class="form-control" id="diffusionRate" min="0.01" max="0.5" step="0.01" value="0.1">
                        <small>Current: <span id="diffusionValue">0.1</span></small>
                    </div>
                </div>
            </div>
            <div class="status-message" id="policyStatus"></div>
            <div class="progress-container" id="policyProgress" style="display: none;">
                <div class="progress-bar" id="policyProgressBar">0%</div>
            </div>
            <div class="capability-actions">
                <button type="submit" class="btn btn-primary">
                    <i class="fas fa-gavel"></i> Run Policy Analysis
                </button>
                <button type="button" class="btn btn-secondary" onclick="closeModal('policyModal')">
                    Cancel
                </button>
            </div>
        </form>
    `);
    
    // Setup policy type change handler
    document.getElementById('policyType').addEventListener('change', function(e) {
        showPolicyOptions(e.target.value);
    });
    
    // Setup sliders
    setupSlider('policyIntensity', 'intensityValue');
    setupSlider('regulationStrength', 'regStrengthValue');
    setupSlider('innovationBoost', 'innovationValue');
    setupSlider('diffusionRate', 'diffusionValue');
    
    // Setup form submission
    document.getElementById('policyForm').addEventListener('submit', handlePolicyAnalysis);
    
    document.body.appendChild(modal);
    modal.style.display = 'block';
}

function showPolicyOptions(policyType) {
    // Hide all option groups
    document.getElementById('monetaryOptions').style.display = 'none';
    document.getElementById('fiscalOptions').style.display = 'none';
    document.getElementById('regulatoryOptions').style.display = 'none';
    document.getElementById('technologyOptions').style.display = 'none';
    
    // Show relevant options
    document.getElementById(policyType + 'Options').style.display = 'block';
}

// Crisis Prediction Modal
function showCrisisModal() {
    const modal = createModal('crisisModal', 'Crisis Prediction Configuration', `
        <form id="crisisForm">
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">Prediction Horizon (Steps)</label>
                    <input type="number" class="form-control" id="predictionHorizon" value="25" min="5" max="100">
                </div>
                <div class="form-group">
                    <label class="form-label">Ensemble Models</label>
                    <select class="form-control" id="ensembleModels" multiple>
                        <option value="neural_network" selected>Neural Network</option>
                        <option value="random_forest" selected>Random Forest</option>
                        <option value="xgboost" selected>XGBoost</option>
                        <option value="lstm" selected>LSTM</option>
                        <option value="svm">Support Vector Machine</option>
                    </select>
                </div>
            </div>
            <div class="form-group">
                <label class="form-label">Risk Thresholds</label>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Volatility Threshold</label>
                        <input type="range" class="form-control" id="volatilityThreshold" min="0.05" max="0.3" step="0.01" value="0.15">
                        <small>Current: <span id="volThresholdValue">0.15</span></small>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Network Fragility</label>
                        <input type="range" class="form-control" id="networkFragility" min="0.3" max="0.9" step="0.05" value="0.7">
                        <small>Current: <span id="fragilityValue">0.7</span></small>
                    </div>
                </div>
            </div>
            <div class="form-group">
                <label class="form-label">
                    <input type="checkbox" id="enableMonitoring" checked> Enable Real-time Monitoring
                </label>
            </div>
            <div class="status-message" id="crisisStatus"></div>
            <div class="progress-container" id="crisisProgress" style="display: none;">
                <div class="progress-bar" id="crisisProgressBar">0%</div>
            </div>
            <div class="capability-actions">
                <button type="submit" class="btn btn-warning">
                    <i class="fas fa-search"></i> Predict Crisis
                </button>
                <button type="button" class="btn btn-secondary" onclick="closeModal('crisisModal')">
                    Cancel
                </button>
            </div>
        </form>
    `);
    
    // Setup sliders
    setupSlider('volatilityThreshold', 'volThresholdValue');
    setupSlider('networkFragility', 'fragilityValue');
    
    // Setup form submission
    document.getElementById('crisisForm').addEventListener('submit', handleCrisisPrediction);
    
    document.body.appendChild(modal);
    modal.style.display = 'block';
}

// Quantum Computing Modal
function showQuantumModal() {
    const modal = createModal('quantumModal', 'Quantum Computing Configuration', `
        <form id="quantumForm">
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">Quantum Backend</label>
                    <select class="form-control" id="quantumBackend">
                        <option value="simulator">Quantum Simulator</option>
                        <option value="ibm">IBM Quantum</option>
                        <option value="rigetti">Rigetti Quantum Cloud</option>
                        <option value="ionq">IonQ</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label">Number of Qubits</label>
                    <input type="number" class="form-control" id="numQubits" value="20" min="4" max="50">
                </div>
            </div>
            <div class="form-group">
                <label class="form-label">Quantum Algorithm</label>
                <select class="form-control" id="quantumAlgorithm">
                    <option value="portfolio">Portfolio Optimization</option>
                    <option value="network">Network Optimization</option>
                    <option value="ml">Quantum Machine Learning</option>
                    <option value="collective">Collective Intelligence</option>
                </select>
            </div>
            <div class="form-group" id="portfolioOptions">
                <label class="form-label">Risk Tolerance</label>
                <input type="range" class="form-control" id="riskTolerance" min="0.1" max="1.0" step="0.1" value="0.5">
                <small>Current: <span id="riskValue">0.5</span></small>
            </div>
            <div class="form-group" id="networkOptions" style="display: none;">
                <label class="form-label">Network Objectives</label>
                <div>
                    <label><input type="checkbox" checked> Efficiency</label>
                    <label><input type="checkbox" checked> Robustness</label>
                    <label><input type="checkbox"> Centrality</label>
                </div>
            </div>
            <div class="form-group" id="mlOptions" style="display: none;">
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">ML Algorithm</label>
                        <select class="form-control" id="qmlAlgorithm">
                            <option value="variational_classifier">Variational Classifier</option>
                            <option value="quantum_svm">Quantum SVM</option>
                            <option value="quantum_neural_network">Quantum Neural Network</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">Circuit Layers</label>
                        <input type="number" class="form-control" id="circuitLayers" value="4" min="1" max="10">
                    </div>
                </div>
            </div>
            <div class="status-message" id="quantumStatus"></div>
            <div class="progress-container" id="quantumProgress" style="display: none;">
                <div class="progress-bar" id="quantumProgressBar">0%</div>
            </div>
            <div class="capability-actions">
                <button type="submit" class="btn btn-primary">
                    <i class="fas fa-magic"></i> Run Quantum Algorithm
                </button>
                <button type="button" class="btn btn-secondary" onclick="closeModal('quantumModal')">
                    Cancel
                </button>
            </div>
        </form>
    `);
    
    // Setup algorithm change handler
    document.getElementById('quantumAlgorithm').addEventListener('change', function(e) {
        showQuantumOptions(e.target.value);
    });
    
    // Setup slider
    setupSlider('riskTolerance', 'riskValue');
    
    // Setup form submission
    document.getElementById('quantumForm').addEventListener('submit', handleQuantumComputation);
    
    document.body.appendChild(modal);
    modal.style.display = 'block';
}

function showQuantumOptions(algorithm) {
    // Hide all option groups
    document.getElementById('portfolioOptions').style.display = 'none';
    document.getElementById('networkOptions').style.display = 'none';
    document.getElementById('mlOptions').style.display = 'none';
    
    // Show relevant options
    if (algorithm === 'portfolio') {
        document.getElementById('portfolioOptions').style.display = 'block';
    } else if (algorithm === 'network') {
        document.getElementById('networkOptions').style.display = 'block';
    } else if (algorithm === 'ml') {
        document.getElementById('mlOptions').style.display = 'block';
    }
}

// Machine Learning Modal
function showMLModal() {
    const modal = createModal('mlModal', 'Machine Learning Configuration', `
        <form id="mlForm">
            <div class="form-group">
                <label class="form-label">Base Models</label>
                <div class="checkbox-grid">
                    <label><input type="checkbox" value="neural_network" checked> Neural Network</label>
                    <label><input type="checkbox" value="random_forest" checked> Random Forest</label>
                    <label><input type="checkbox" value="xgboost" checked> XGBoost</label>
                    <label><input type="checkbox" value="lstm" checked> LSTM</label>
                    <label><input type="checkbox" value="svm"> Support Vector Machine</label>
                    <label><input type="checkbox" value="elastic_net"> Elastic Net</label>
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label class="form-label">Ensemble Method</label>
                    <select class="form-control" id="ensembleMethod">
                        <option value="stacking">Stacking</option>
                        <option value="voting">Voting</option>
                        <option value="boosting">Boosting</option>
                        <option value="bagging">Bagging</option>
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label">Target Variable</label>
                    <select class="form-control" id="targetVariable">
                        <option value="returns">Returns</option>
                        <option value="volatility">Volatility</option>
                        <option value="crisis_probability">Crisis Probability</option>
                    </select>
                </div>
            </div>
            <div class="form-group">
                <label class="form-label">Training Configuration</label>
                <div class="form-row">
                    <div class="form-group">
                        <label class="form-label">Max Iterations</label>
                        <input type="number" class="form-control" id="maxIterations" value="100" min="10" max="1000">
                    </div>
                    <div class="form-group">
                        <label class="form-label">Cross-Validation Folds</label>
                        <input type="number" class="form-control" id="cvFolds" value="5" min="3" max="10">
                    </div>
                </div>
            </div>
            <div class="form-group">
                <label class="form-label">
                    <input type="checkbox" id="autoHyperparameter" checked> Automatic Hyperparameter Optimization
                </label>
            </div>
            <div class="status-message" id="mlStatus"></div>
            <div class="progress-container" id="mlProgress" style="display: none;">
                <div class="progress-bar" id="mlProgressBar">0%</div>
            </div>
            <div class="capability-actions">
                <button type="submit" class="btn btn-primary">
                    <i class="fas fa-cogs"></i> Train ML Ensemble
                </button>
                <button type="button" class="btn btn-secondary" onclick="closeModal('mlModal')">
                    Cancel
                </button>
            </div>
        </form>
    `);
    
    // Setup form submission
    document.getElementById('mlForm').addEventListener('submit', handleMLTraining);
    
    document.body.appendChild(modal);
    modal.style.display = 'block';
}

// System Status Modal
function showSystemStatus() {
    const modal = createModal('statusModal', 'System Status Dashboard', `
        <div class="status-dashboard">
            <div class="status-grid">
                <div class="status-section">
                    <h4><i class="fas fa-server"></i> Services</h4>
                    <div class="service-status">
                        <div class="service-item">
                            <span>API Server</span>
                            <span class="status-badge status-online">Online</span>
                        </div>
                        <div class="service-item">
                            <span>Database</span>
                            <span class="status-badge status-online">Online</span>
                        </div>
                        <div class="service-item">
                            <span>Redis Cache</span>
                            <span class="status-badge status-online">Online</span>
                        </div>
                        <div class="service-item">
                            <span>Streaming Service</span>
                            <span class="status-badge status-offline">Offline</span>
                        </div>
                        <div class="service-item">
                            <span>Quantum Service</span>
                            <span class="status-badge status-offline">Offline</span>
                        </div>
                    </div>
                </div>
                <div class="status-section">
                    <h4><i class="fas fa-chart-line"></i> Performance Metrics</h4>
                    <div class="metric-item">
                        <span>CPU Usage</span>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: 35%;"></div>
                            <span class="metric-value">35%</span>
                        </div>
                    </div>
                    <div class="metric-item">
                        <span>Memory Usage</span>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: 68%;"></div>
                            <span class="metric-value">68%</span>
                        </div>
                    </div>
                    <div class="metric-item">
                        <span>Disk Usage</span>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: 45%;"></div>
                            <span class="metric-value">45%</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="recent-activity">
                <h4><i class="fas fa-history"></i> Recent Activity</h4>
                <div class="activity-list">
                    <div class="activity-item">
                        <span class="activity-time">2 min ago</span>
                        <span class="activity-desc">Simulation completed: 1000 agents, 250 steps</span>
                    </div>
                    <div class="activity-item">
                        <span class="activity-time">5 min ago</span>
                        <span class="activity-desc">Policy analysis started: Monetary policy</span>
                    </div>
                    <div class="activity-item">
                        <span class="activity-time">8 min ago</span>
                        <span class="activity-desc">Real-time streaming initiated</span>
                    </div>
                </div>
            </div>
        </div>
        <div class="capability-actions" style="margin-top: 2rem;">
            <button type="button" class="btn btn-primary" onclick="refreshSystemStatus()">
                <i class="fas fa-sync"></i> Refresh Status
            </button>
            <button type="button" class="btn btn-secondary" onclick="closeModal('statusModal')">
                Close
            </button>
        </div>
    `);
    
    document.body.appendChild(modal);
    modal.style.display = 'block';
}

// Logs Modal
function showLogsModal() {
    const modal = createModal('logsModal', 'System Logs', `
        <div class="logs-container">
            <div class="logs-controls">
                <select class="form-control" id="logLevel" style="width: auto; display: inline-block;">
                    <option value="all">All Levels</option>
                    <option value="error">Error</option>
                    <option value="warning">Warning</option>
                    <option value="info">Info</option>
                    <option value="debug">Debug</option>
                </select>
                <select class="form-control" id="logService" style="width: auto; display: inline-block;">
                    <option value="all">All Services</option>
                    <option value="api">API Server</option>
                    <option value="streaming">Streaming</option>
                    <option value="quantum">Quantum</option>
                    <option value="database">Database</option>
                </select>
                <button class="btn btn-secondary" onclick="refreshLogs()">
                    <i class="fas fa-sync"></i> Refresh
                </button>
            </div>
            <div class="logs-content" id="logsContent">
                <div class="log-entry log-info">
                    <span class="log-time">2025-01-27 10:30:15</span>
                    <span class="log-level">INFO</span>
                    <span class="log-service">API</span>
                    <span class="log-message">Simulation started with 1000 agents</span>
                </div>
                <div class="log-entry log-warning">
                    <span class="log-time">2025-01-27 10:29:42</span>
                    <span class="log-level">WARN</span>
                    <span class="log-service">STREAM</span>
                    <span class="log-message">Connection timeout to data source</span>
                </div>
                <div class="log-entry log-info">
                    <span class="log-time">2025-01-27 10:28:30</span>
                    <span class="log-level">INFO</span>
                    <span class="log-service">API</span>
                    <span class="log-message">Economy created successfully</span>
                </div>
            </div>
        </div>
        <div class="capability-actions" style="margin-top: 1rem;">
            <button type="button" class="btn btn-primary" onclick="downloadLogs()">
                <i class="fas fa-download"></i> Download Logs
            </button>
            <button type="button" class="btn btn-secondary" onclick="closeModal('logsModal')">
                Close
            </button>
        </div>
    `);
    
    document.body.appendChild(modal);
    modal.style.display = 'block';
}

// Utility Functions
function createModal(id, title, content) {
    const modal = document.createElement('div');
    modal.id = id;
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title">${title}</h2>
                <span class="close" onclick="closeModal('${id}')">&times;</span>
            </div>
            ${content}
        </div>
    `;
    return modal;
}

function setupSlider(sliderId, valueId) {
    const slider = document.getElementById(sliderId);
    const valueSpan = document.getElementById(valueId);
    
    if (slider && valueSpan) {
        slider.addEventListener('input', function(e) {
            valueSpan.textContent = e.target.value;
        });
    }
}

// Event Handlers
async function handlePolicyAnalysis(e) {
    e.preventDefault();
    
    const policyType = document.getElementById('policyType').value;
    const formData = {
        policy_type: policyType,
        intensity: parseFloat(document.getElementById('policyIntensity').value),
        intervention_timing: parseInt(document.getElementById('interventionTiming').value),
        duration: parseInt(document.getElementById('policyDuration').value)
    };
    
    // Add policy-specific parameters
    if (policyType === 'monetary') {
        formData.interest_rate_change = parseInt(document.getElementById('interestRateChange').value);
    } else if (policyType === 'fiscal') {
        formData.spending_change = parseFloat(document.getElementById('spendingChange').value);
        formData.tax_change = parseFloat(document.getElementById('taxChange').value);
    } else if (policyType === 'regulatory') {
        formData.regulation_strength = parseFloat(document.getElementById('regulationStrength').value);
        formData.scope = document.getElementById('regulationScope').value;
    } else if (policyType === 'technology') {
        formData.innovation_boost = parseFloat(document.getElementById('innovationBoost').value);
        formData.diffusion_rate = parseFloat(document.getElementById('diffusionRate').value);
    }
    
    showStatus('policyStatus', 'Running policy analysis...', 'warning');
    showProgress('policyProgress', 'policyProgressBar');
    
    try {
        // Simulate progress
        for (let i = 0; i <= 100; i += 20) {
            await new Promise(resolve => setTimeout(resolve, 500));
            updateProgress('policyProgressBar', i);
        }
        
        showStatus('policyStatus', 'Policy analysis completed successfully!', 'success');
        
        setTimeout(() => {
            closeModal('policyModal');
            openDashboard('policy');
        }, 2000);
        
    } catch (error) {
        showStatus('policyStatus', 'Policy analysis failed: ' + error.message, 'error');
    }
}

async function handleCrisisPrediction(e) {
    e.preventDefault();
    
    const formData = {
        prediction_horizon: parseInt(document.getElementById('predictionHorizon').value),
        ensemble_models: Array.from(document.getElementById('ensembleModels').selectedOptions).map(o => o.value),
        risk_thresholds: {
            volatility: parseFloat(document.getElementById('volatilityThreshold').value),
            network_fragility: parseFloat(document.getElementById('networkFragility').value)
        },
        enable_monitoring: document.getElementById('enableMonitoring').checked
    };
    
    showStatus('crisisStatus', 'Running crisis prediction...', 'warning');
    showProgress('crisisProgress', 'crisisProgressBar');
    
    try {
        // Simulate progress
        for (let i = 0; i <= 100; i += 25) {
            await new Promise(resolve => setTimeout(resolve, 800));
            updateProgress('crisisProgressBar', i);
        }
        
        showStatus('crisisStatus', 'Crisis prediction completed!', 'success');
        
        setTimeout(() => {
            closeModal('crisisModal');
            openDashboard('risk');
        }, 2000);
        
    } catch (error) {
        showStatus('crisisStatus', 'Crisis prediction failed: ' + error.message, 'error');
    }
}

async function handleQuantumComputation(e) {
    e.preventDefault();
    
    const algorithm = document.getElementById('quantumAlgorithm').value;
    const formData = {
        backend: document.getElementById('quantumBackend').value,
        n_qubits: parseInt(document.getElementById('numQubits').value),
        algorithm: algorithm
    };
    
    // Add algorithm-specific parameters
    if (algorithm === 'portfolio') {
        formData.risk_tolerance = parseFloat(document.getElementById('riskTolerance').value);
    }
    
    showStatus('quantumStatus', 'Initializing quantum computation...', 'warning');
    showProgress('quantumProgress', 'quantumProgressBar');
    
    try {
        // Update quantum status
        updateStatusIndicator('quantum-status', 'warning');
        
        // Simulate progress
        for (let i = 0; i <= 100; i += 10) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            updateProgress('quantumProgressBar', i);
        }
        
        showStatus('quantumStatus', 'Quantum computation completed!', 'success');
        updateStatusIndicator('quantum-status', 'online');
        systemStats.quantumJobs++;
        updateStats();
        
        setTimeout(() => {
            closeModal('quantumModal');
            openDashboard('quantum');
        }, 2000);
        
    } catch (error) {
        showStatus('quantumStatus', 'Quantum computation failed: ' + error.message, 'error');
        updateStatusIndicator('quantum-status', 'offline');
    }
}

async function handleMLTraining(e) {
    e.preventDefault();
    
    const baseModels = Array.from(document.querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.value);
    const formData = {
        base_models: baseModels,
        ensemble_method: document.getElementById('ensembleMethod').value,
        target_variable: document.getElementById('targetVariable').value,
        max_iterations: parseInt(document.getElementById('maxIterations').value),
        cv_folds: parseInt(document.getElementById('cvFolds').value),
        auto_hyperparameter: document.getElementById('autoHyperparameter').checked
    };
    
    showStatus('mlStatus', 'Training ML ensemble...', 'warning');
    showProgress('mlProgress', 'mlProgressBar');
    
    try {
        // Simulate training progress
        const steps = ['Preparing data', 'Training models', 'Optimizing hyperparameters', 'Evaluating ensemble'];
        for (let i = 0; i < steps.length; i++) {
            showStatus('mlStatus', steps[i] + '...', 'warning');
            await new Promise(resolve => setTimeout(resolve, 2000));
            updateProgress('mlProgressBar', ((i + 1) / steps.length) * 100);
        }
        
        showStatus('mlStatus', 'ML ensemble training completed!', 'success');
        
        setTimeout(() => {
            closeModal('mlModal');
            openDashboard('ml');
        }, 2000);
        
    } catch (error) {
        showStatus('mlStatus', 'ML training failed: ' + error.message, 'error');
    }
}

function refreshSystemStatus() {
    // Refresh system status data
    checkSystemStatus();
    showStatus('System status refreshed', '', 'success');
}

function refreshLogs() {
    // Refresh logs content
    const logsContent = document.getElementById('logsContent');
    if (logsContent) {
        // Add loading indicator
        logsContent.innerHTML = '<div class="loading"></div> Loading logs...';
        
        // Simulate log refresh
        setTimeout(() => {
            logsContent.innerHTML = `
                <div class="log-entry log-info">
                    <span class="log-time">${new Date().toLocaleString()}</span>
                    <span class="log-level">INFO</span>
                    <span class="log-service">API</span>
                    <span class="log-message">Logs refreshed</span>
                </div>
            `;
        }, 1000);
    }
}

function downloadLogs() {
    // Create and download logs file
    const logs = document.getElementById('logsContent').innerText;
    const blob = new Blob([logs], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `neuricx-logs-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}