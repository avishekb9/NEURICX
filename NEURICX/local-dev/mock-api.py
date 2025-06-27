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

@app.route('/simulation/<sim_id>/status')
def get_simulation_status(sim_id):
    if sim_id not in jobs:
        return jsonify({"error": "Simulation not found"}), 404
    
    return jsonify({
        "simulation_id": sim_id,
        "status": jobs[sim_id]["status"],
        "progress": jobs[sim_id]["progress"],
        "completed": jobs[sim_id]["status"] == "completed",
        "message": f"Simulation {jobs[sim_id]['status']} ({jobs[sim_id]['progress']}%)"
    })

@app.route('/streaming/start', methods=['POST'])
def start_streaming():
    return jsonify({
        "stream_id": f"stream_{int(time.time())}",
        "status": "started",
        "message": "Real-time streaming initiated"
    })

@app.route('/policy/analyze', methods=['POST'])
def analyze_policy():
    data = request.json
    policy_id = f"policy_{int(time.time())}_{random.randint(1000, 9999)}"
    
    return jsonify({
        "analysis_id": policy_id,
        "policy_type": data.get('policy_type', 'monetary'),
        "intervention_magnitude": data.get('intervention_magnitude', 0.25),
        "status": "completed",
        "results": {
            "network_transmission_effect": random.uniform(0.6, 0.9),
            "agent_impact_scores": {
                "household": random.uniform(0.3, 0.7),
                "firm": random.uniform(0.4, 0.8),
                "bank": random.uniform(0.8, 1.2),
                "government": random.uniform(0.1, 0.3),
                "central_bank": random.uniform(1.5, 2.0),
                "foreign": random.uniform(0.2, 0.5)
            }
        }
    })

@app.route('/crisis/predict', methods=['POST'])
def predict_crisis():
    data = request.json
    crisis_id = f"crisis_{int(time.time())}_{random.randint(1000, 9999)}"
    
    return jsonify({
        "prediction_id": crisis_id,
        "status": "completed",
        "results": {
            "crisis_probability": random.uniform(0.1, 0.8),
            "risk_level": random.choice(["low", "medium", "high"]),
            "predicted_severity": random.uniform(0.2, 0.9),
            "time_horizon": data.get('prediction_horizon', 90),
            "early_warning_indicators": [
                {"indicator": "credit_growth", "value": random.uniform(-0.1, 0.3), "threshold": 0.2},
                {"indicator": "asset_prices", "value": random.uniform(-0.2, 0.4), "threshold": 0.3},
                {"indicator": "leverage_ratio", "value": random.uniform(0.1, 0.6), "threshold": 0.4}
            ]
        }
    })

@app.route('/quantum/initialize', methods=['POST'])
def initialize_quantum():
    data = request.json
    quantum_id = f"quantum_{int(time.time())}_{random.randint(1000, 9999)}"
    
    return jsonify({
        "quantum_id": quantum_id,
        "backend": data.get('backend', 'qasm_simulator'),
        "n_qubits": data.get('n_qubits', 4),
        "status": "initialized"
    })

@app.route('/quantum/run', methods=['POST'])
def run_quantum():
    data = request.json
    quantum_id = f"quantum_{int(time.time())}_{random.randint(1000, 9999)}"
    
    return jsonify({
        "quantum_id": quantum_id,
        "algorithm": data.get('algorithm', 'portfolio'),
        "status": "completed",
        "results": {
            "quantum_advantage": random.uniform(1.1, 2.5),
            "execution_time": random.uniform(0.5, 5.0),
            "fidelity": random.uniform(0.85, 0.99),
            "optimization_result": {
                "cost_reduction": random.uniform(0.1, 0.4),
                "solution_quality": random.uniform(0.8, 0.95)
            }
        }
    })

@app.route('/ml/train', methods=['POST'])
def train_ml_ensemble():
    data = request.json
    ml_id = f"ml_{int(time.time())}_{random.randint(1000, 9999)}"
    
    return jsonify({
        "ml_id": ml_id,
        "ensemble_method": data.get('ensemble_method', 'voting'),
        "status": "completed", 
        "results": {
            "accuracy": random.uniform(0.75, 0.95),
            "precision": random.uniform(0.70, 0.90),
            "recall": random.uniform(0.72, 0.88),
            "f1_score": random.uniform(0.74, 0.89),
            "models_trained": data.get('base_models', ['rf', 'xgb', 'svm', 'nn'])
        }
    })

if __name__ == '__main__':
    print("🔌 Starting NEURICX Mock API Server...")
    print("📡 API available at http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)