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

@app.route('/sisir/analyze', methods=['POST'])
def sisir_analyze():
    data = request.json
    sisir_id = f"sisir_{int(time.time())}_{random.randint(1000, 9999)}"
    
    platforms = data.get('platforms', ['instagram'])
    accounts = data.get('accounts', {})
    analysis_type = data.get('analysis_type', 'comprehensive')
    
    # Generate realistic SISIR analysis results
    profile_data = {
        'follower_count': random.randint(5000, 50000),
        'following_count': random.randint(500, 5000),
        'post_count': random.randint(100, 2000),
        'verified': random.choice([True, False]),
        'engagement_rate': random.uniform(2.5, 8.5)
    }
    
    economic_impact = {
        'overall_score': random.uniform(65, 95),
        'market_influence': random.uniform(60, 90),
        'consumer_sentiment_impact': random.uniform(70, 95),
        'brand_value_correlation': random.uniform(65, 85),
        'viral_potential': random.uniform(55, 90),
        'economic_keyword_density': random.uniform(40, 80)
    }
    
    viral_predictions = {
        'top_viral_hashtags': {
            '#finance': 15,
            '#investing': 12,
            '#crypto': 8,
            '#stocks': 7,
            '#economy': 5
        },
        'optimal_content_length': random.randint(80, 280),
        'optimal_posting_hour': random.randint(9, 21),
        'engagement_threshold': random.uniform(3.0, 7.0)
    }
    
    growth_strategies = {
        'content_optimization': {
            'recommended_posting_frequency': '1-2 posts per day',
            'best_content_types': ['market analysis', 'educational posts', 'behind-the-scenes'],
            'hashtag_strategy': 'Mix of 8-12 hashtags: trending + niche + branded'
        },
        'audience_growth': {
            'target_demographics': ['25-45 professionals', 'finance enthusiasts', 'entrepreneurs'],
            'collaboration_opportunities': ['fintech influencers', 'market analysts', 'startup founders']
        },
        'economic_positioning': {
            'market_alignment': 'Focus on data-driven insights',
            'trending_topics': ['AI in finance', 'sustainable investing', 'DeFi trends']
        }
    }
    
    market_correlations = {
        'sp500_correlation': random.uniform(0.3, 0.8),
        'crypto_correlation': random.uniform(0.2, 0.7),
        'vix_correlation': random.uniform(-0.6, -0.2),
        'consumer_confidence_correlation': random.uniform(0.4, 0.9)
    }
    
    return jsonify({
        "sisir_id": sisir_id,
        "analysis_type": analysis_type,
        "platforms": platforms,
        "status": "completed",
        "profile_data": profile_data,
        "economic_impact": economic_impact,
        "viral_predictions": viral_predictions,
        "growth_strategies": growth_strategies,
        "market_correlations": market_correlations,
        "sentiment_analysis": {
            "average_sentiment": random.uniform(-0.2, 0.8),
            "sentiment_trend": random.choice(['improving', 'stable', 'declining']),
            "economic_sentiment": random.uniform(0.3, 0.9),
            "confidence_score": random.uniform(0.7, 0.95)
        },
        "recommendations": [
            "Increase focus on educational content about market trends",
            "Optimize posting times based on audience engagement patterns",
            "Collaborate with fintech influencers to expand reach",
            "Leverage trending economic topics for viral potential",
            "Build community through interactive Q&A sessions"
        ],
        "economic_forecast": {
            "sentiment_forecast": random.choice([
                "Improving market sentiment - potential upward trend",
                "Stable market sentiment - sideways movement expected",
                "Mixed signals - monitor for trend confirmation"
            ]),
            "confidence_level": random.uniform(75, 95),
            "key_indicators": {
                "social_sentiment_score": random.uniform(0.2, 0.8),
                "sentiment_momentum": random.uniform(-0.05, 0.05),
                "market_correlation_strength": random.uniform(0.4, 0.8)
            }
        }
    })

@app.route('/sisir/sentiment', methods=['POST'])
def sisir_sentiment_stream():
    data = request.json
    stream_id = f"sentiment_stream_{int(time.time())}_{random.randint(1000, 9999)}"
    
    return jsonify({
        "stream_id": stream_id,
        "status": "started",
        "message": "Real-time sentiment monitoring initiated",
        "platforms": data.get('platforms', ['instagram', 'twitter']),
        "update_frequency": data.get('update_frequency', 300),  # seconds
        "current_sentiment": {
            "overall_score": random.uniform(-0.3, 0.7),
            "market_correlation": random.uniform(0.3, 0.8),
            "trending_topics": ['AI investment', 'crypto adoption', 'sustainable finance'],
            "sentiment_by_platform": {
                "instagram": random.uniform(0.1, 0.8),
                "twitter": random.uniform(-0.2, 0.6),
                "facebook": random.uniform(0.0, 0.5)
            }
        }
    })

@app.route('/sisir/viral-prediction', methods=['POST'])
def sisir_viral_prediction():
    data = request.json
    content = data.get('content', '')
    
    # Mock viral prediction based on content analysis
    score = random.uniform(0.2, 0.95)
    
    factors = {
        'content_quality': random.uniform(0.6, 0.9),
        'timing_optimization': random.uniform(0.5, 0.8),
        'hashtag_effectiveness': random.uniform(0.4, 0.9),
        'audience_alignment': random.uniform(0.7, 0.95),
        'trend_relevance': random.uniform(0.3, 0.8)
    }
    
    return jsonify({
        "viral_probability": score,
        "prediction_confidence": random.uniform(0.75, 0.95),
        "factors": factors,
        "recommendations": [
            "Add trending hashtags #fintech #investing",
            "Post during peak hours (2-4 PM)",
            "Include a call-to-action question",
            "Use engaging visuals or charts",
            "Tag relevant industry influencers"
        ],
        "estimated_reach": {
            "base_reach": random.randint(1000, 5000),
            "viral_multiplier": round(score * 10, 1),
            "potential_reach": random.randint(10000, 100000)
        }
    })

if __name__ == '__main__':
    print("🔌 Starting NEURICX Mock API Server...")
    print("📡 API available at http://localhost:8000")
    app.run(host='0.0.0.0', port=8000, debug=True)