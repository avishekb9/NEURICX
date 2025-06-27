// NEURICX Platform Demo Mode - GitHub Pages Compatible
// This script provides demo functionality without requiring backend services

class NEURICXDemo {
    constructor() {
        this.demoMode = true;
        this.activeJobs = new Map();
        this.systemMetrics = {
            uptime: 0,
            activeSimulations: 0,
            totalAgents: 0,
            quantumJobs: 0,
            memoryUsage: 45.2,
            cpuUsage: 23.7
        };
        
        this.startTime = Date.now();
        this.initialize();
    }

    initialize() {
        console.log('🚀 Initializing NEURICX Demo Platform...');
        this.startDemoMode();
        this.updateSystemMetrics();
        setInterval(() => this.updateSystemMetrics(), 5000);
    }

    startDemoMode() {
        // Show demo mode notification
        const demoAlert = document.createElement('div');
        demoAlert.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ff6b35;
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            z-index: 10000;
            font-family: 'Exo 2', sans-serif;
            box-shadow: 0 5px 15px rgba(255, 107, 53, 0.3);
            animation: slideIn 0.5s ease-out;
        `;
        demoAlert.innerHTML = `
            <strong>🎯 DEMO MODE</strong><br>
            <small>Running simulation demos without backend</small>
        `;
        document.body.appendChild(demoAlert);

        // Add animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);

        // Auto-hide after 5 seconds
        setTimeout(() => {
            demoAlert.style.animation = 'slideIn 0.5s ease-out reverse';
            setTimeout(() => demoAlert.remove(), 500);
        }, 5000);
    }

    // Demo simulation
    async createSimulation(config) {
        console.log('🧠 Running demo simulation...', config);
        
        const simulationId = this.generateId('demo_sim');
        this.activeJobs.set(simulationId, {
            type: 'simulation',
            status: 'running',
            progress: 0,
            config: config,
            startTime: Date.now()
        });

        try {
            const steps = [
                { progress: 10, message: 'Creating economic environment...' },
                { progress: 25, message: 'Initializing 1000 agents...' },
                { progress: 40, message: 'Building network connections...' },
                { progress: 55, message: 'Running simulation steps...' },
                { progress: 70, message: 'Analyzing agent interactions...' },
                { progress: 85, message: 'Computing network metrics...' },
                { progress: 100, message: 'Simulation completed!' }
            ];

            for (const step of steps) {
                this.updateJobStatus(simulationId, 'running', step.progress, step.message);
                await this.delay(800 + Math.random() * 1200);
            }

            this.updateJobStatus(simulationId, 'completed', 100, 'Demo simulation completed');
            this.systemMetrics.activeSimulations++;
            this.systemMetrics.totalAgents += config.numAgents || 1000;

            return {
                simulationId,
                status: 'completed',
                results: {
                    final_collective_intelligence: 1.15 + Math.random() * 0.2 - 0.1,
                    average_wealth: 10000 + Math.random() * 2000 - 1000,
                    wealth_inequality: 0.3 + Math.random() * 0.1 - 0.05,
                    network_density: config.networkDensity || 0.05,
                    simulation_steps: config.simulationSteps || 250
                }
            };

        } catch (error) {
            this.updateJobStatus(simulationId, 'failed', 0, error.message);
            throw error;
        }
    }

    // Demo streaming
    async startStreaming(config) {
        console.log('📡 Starting demo streaming...', config);
        
        const streamId = this.generateId('demo_stream');
        this.activeJobs.set(streamId, {
            type: 'streaming',
            status: 'running',
            progress: 100,
            config: config,
            startTime: Date.now()
        });

        this.updateJobStatus(streamId, 'completed', 100, 'Demo streaming started');
        
        // Simulate real-time data updates
        this.startDataSimulation();

        return {
            streamId,
            status: 'started',
            message: 'Demo real-time data simulation active'
        };
    }

    startDataSimulation() {
        // Simulate live data updates every 3 seconds
        setInterval(() => {
            const mockData = {
                timestamp: Date.now(),
                market_data: {
                    'AAPL': 150 + Math.random() * 20 - 10,
                    'MSFT': 300 + Math.random() * 30 - 15,
                    'GOOGL': 2500 + Math.random() * 200 - 100
                },
                economic_indicators: {
                    inflation: 2.1 + Math.random() * 0.5 - 0.25,
                    unemployment: 3.7 + Math.random() * 0.3 - 0.15,
                    gdp_growth: 2.3 + Math.random() * 0.4 - 0.2
                }
            };
            
            console.log('📊 Demo data update:', mockData);
            
            // Emit event for UI updates
            this.emit('dataUpdate', mockData);
        }, 3000);
    }

    // Demo policy analysis
    async runPolicyAnalysis(config) {
        console.log('⚖️ Running demo policy analysis...', config);
        
        const analysisId = this.generateId('demo_policy');
        this.activeJobs.set(analysisId, {
            type: 'policy',
            status: 'running',
            progress: 0,
            config: config,
            startTime: Date.now()
        });

        try {
            const steps = [
                { progress: 20, message: 'Preparing policy scenario...' },
                { progress: 40, message: 'Running baseline simulation...' },
                { progress: 60, message: 'Applying policy intervention...' },
                { progress: 80, message: 'Analyzing transmission effects...' },
                { progress: 100, message: 'Policy analysis completed' }
            ];

            for (const step of steps) {
                this.updateJobStatus(analysisId, 'running', step.progress, step.message);
                await this.delay(1000 + Math.random() * 1000);
            }

            this.updateJobStatus(analysisId, 'completed', 100, 'Demo policy analysis completed');

            return {
                analysisId,
                status: 'completed',
                results: {
                    policy_type: config.policyType || 'monetary',
                    intervention_magnitude: config.interventionMagnitude || 0.25,
                    network_transmission_effect: 0.6 + Math.random() * 0.3,
                    agent_impact_scores: {
                        household: 0.3 + Math.random() * 0.4,
                        firm: 0.4 + Math.random() * 0.4,
                        bank: 0.8 + Math.random() * 0.4,
                        government: 0.1 + Math.random() * 0.2,
                        central_bank: 1.5 + Math.random() * 0.5,
                        foreign: 0.2 + Math.random() * 0.3
                    }
                }
            };

        } catch (error) {
            this.updateJobStatus(analysisId, 'failed', 0, error.message);
            throw error;
        }
    }

    // Demo crisis prediction
    async runCrisisPrediction(config) {
        console.log('⚠️ Running demo crisis prediction...', config);
        
        const predictionId = this.generateId('demo_crisis');
        this.activeJobs.set(predictionId, {
            type: 'crisis',
            status: 'running',
            progress: 0,
            config: config,
            startTime: Date.now()
        });

        try {
            const steps = [
                { progress: 15, message: 'Extracting features...' },
                { progress: 30, message: 'Training neural network...' },
                { progress: 45, message: 'Training random forest...' },
                { progress: 60, message: 'Training XGBoost...' },
                { progress: 75, message: 'Training LSTM...' },
                { progress: 90, message: 'Combining ensemble predictions...' },
                { progress: 100, message: 'Crisis prediction completed' }
            ];

            for (const step of steps) {
                this.updateJobStatus(predictionId, 'running', step.progress, step.message);
                await this.delay(800 + Math.random() * 1200);
            }

            this.updateJobStatus(predictionId, 'completed', 100, 'Demo crisis prediction completed');

            return {
                predictionId,
                status: 'completed',
                results: {
                    crisis_probability: Math.random() * 0.7 + 0.1,
                    risk_level: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
                    predicted_severity: Math.random() * 0.7 + 0.2,
                    time_horizon: config.predictionHorizon || 90,
                    early_warning_indicators: [
                        { indicator: 'credit_growth', value: Math.random() * 0.4 - 0.1, threshold: 0.2 },
                        { indicator: 'asset_prices', value: Math.random() * 0.6 - 0.2, threshold: 0.3 },
                        { indicator: 'leverage_ratio', value: Math.random() * 0.5 + 0.1, threshold: 0.4 }
                    ]
                }
            };

        } catch (error) {
            this.updateJobStatus(predictionId, 'failed', 0, error.message);
            throw error;
        }
    }

    // Demo quantum computing
    async runQuantumComputation(config) {
        console.log('⚛️ Running demo quantum computation...', config);
        
        const quantumId = this.generateId('demo_quantum');
        this.activeJobs.set(quantumId, {
            type: 'quantum',
            status: 'running',
            progress: 0,
            config: config,
            startTime: Date.now()
        });

        try {
            const steps = [
                { progress: 20, message: 'Initializing quantum simulator...' },
                { progress: 40, message: 'Preparing quantum circuits...' },
                { progress: 60, message: 'Executing quantum algorithm...' },
                { progress: 80, message: 'Processing quantum results...' },
                { progress: 100, message: 'Quantum computation completed' }
            ];

            for (const step of steps) {
                this.updateJobStatus(quantumId, 'running', step.progress, step.message);
                await this.delay(1000 + Math.random() * 2000);
            }

            this.updateJobStatus(quantumId, 'completed', 100, 'Demo quantum computation completed');
            this.systemMetrics.quantumJobs++;

            return {
                quantumId,
                status: 'completed',
                results: {
                    algorithm: config.algorithm || 'portfolio',
                    backend: config.backend || 'qasm_simulator',
                    n_qubits: config.numQubits || 4,
                    quantum_advantage: 1.1 + Math.random() * 1.4,
                    execution_time: Math.random() * 4.5 + 0.5,
                    fidelity: 0.85 + Math.random() * 0.14,
                    optimization_result: {
                        cost_reduction: Math.random() * 0.3 + 0.1,
                        solution_quality: 0.8 + Math.random() * 0.15
                    }
                }
            };

        } catch (error) {
            this.updateJobStatus(quantumId, 'failed', 0, error.message);
            throw error;
        }
    }

    // Demo ML training
    async trainMLEnsemble(config) {
        console.log('🤖 Training demo ML ensemble...', config);
        
        const mlId = this.generateId('demo_ml');
        this.activeJobs.set(mlId, {
            type: 'ml',
            status: 'running',
            progress: 0,
            config: config,
            startTime: Date.now()
        });

        try {
            const steps = [
                { progress: 15, message: 'Preparing training data...' },
                { progress: 30, message: 'Training Random Forest...' },
                { progress: 45, message: 'Training XGBoost...' },
                { progress: 60, message: 'Training Neural Network...' },
                { progress: 75, message: 'Training SVM...' },
                { progress: 90, message: 'Creating ensemble...' },
                { progress: 100, message: 'ML training completed' }
            ];

            for (const step of steps) {
                this.updateJobStatus(mlId, 'running', step.progress, step.message);
                await this.delay(700 + Math.random() * 1000);
            }

            this.updateJobStatus(mlId, 'completed', 100, 'Demo ML ensemble training completed');

            return {
                mlId,
                status: 'completed',
                results: {
                    ensemble_method: config.ensembleMethod || 'voting',
                    accuracy: 0.75 + Math.random() * 0.2,
                    precision: 0.70 + Math.random() * 0.2,
                    recall: 0.72 + Math.random() * 0.16,
                    f1_score: 0.74 + Math.random() * 0.15,
                    models_trained: config.baseModels || ['rf', 'xgb', 'svm', 'nn']
                }
            };

        } catch (error) {
            this.updateJobStatus(mlId, 'failed', 0, error.message);
            throw error;
        }
    }

    // Utility methods
    generateId(prefix) {
        return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    updateJobStatus(jobId, status, progress, message) {
        if (this.activeJobs.has(jobId)) {
            const job = this.activeJobs.get(jobId);
            job.status = status;
            job.progress = progress;
            job.message = message;
            
            console.log(`📊 Demo Job ${jobId}: ${status} (${progress}%) - ${message}`);
            
            // Emit event for UI updates
            this.emit('jobUpdate', { jobId, status, progress, message });
        }
    }

    updateSystemMetrics() {
        this.systemMetrics.uptime = Math.floor((Date.now() - this.startTime) / 1000);
        this.systemMetrics.memoryUsage = 40 + Math.random() * 20;
        this.systemMetrics.cpuUsage = 15 + Math.random() * 30;
        
        // Update UI
        if (typeof updateStats === 'function') {
            updateStats(this.systemMetrics);
        }
    }

    // Event emitter functionality
    emit(event, data) {
        if (typeof window !== 'undefined') {
            window.dispatchEvent(new CustomEvent('neuricx-' + event, { detail: data }));
        }
    }

    async delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize demo when DOM is loaded
if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', () => {
        window.neuricxDemo = new NEURICXDemo();
    });
}