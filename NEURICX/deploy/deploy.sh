#!/bin/bash

# NEURICX Deployment Script
# Comprehensive deployment automation for NEURICX framework

set -e  # Exit on any error

# Color codes for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

# Print colored output
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

# Configuration
DEPLOYMENT_ENV=${1:-"development"}
DOMAIN=${2:-"localhost"}
SSL_ENABLED=${3:-"false"}

print_status "Starting NEURICX deployment for environment: $DEPLOYMENT_ENV"

# Validate environment
if [[ ! "$DEPLOYMENT_ENV" =~ ^(development|staging|production)$ ]]; then
    print_error "Invalid environment. Must be one of: development, staging, production"
    exit 1
fi

# Check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "All dependencies are available"
}

# Setup environment variables
setup_environment() {
    print_status "Setting up environment variables..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f .env ]]; then
        print_status "Creating .env file..."
        cat > .env << EOF
# NEURICX Environment Configuration
DEPLOYMENT_ENV=$DEPLOYMENT_ENV
DOMAIN=$DOMAIN
SSL_ENABLED=$SSL_ENABLED

# Database Configuration
POSTGRES_DB=neuricx
POSTGRES_USER=neuricx
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# Redis Configuration  
REDIS_PASSWORD=$(openssl rand -base64 16)

# API Configuration
NEURICX_API_SECRET=$(openssl rand -base64 32)
JUPYTER_TOKEN=$(openssl rand -base64 16)

# Quantum Computing API Keys (optional)
QISKIT_IBM_TOKEN=${QISKIT_IBM_TOKEN:-""}
IONQ_API_TOKEN=${IONQ_API_TOKEN:-""}

# Monitoring
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 16)

# External API Keys (for real-time data)
ALPHA_VANTAGE_API_KEY=${ALPHA_VANTAGE_API_KEY:-""}
POLYGON_API_KEY=${POLYGON_API_KEY:-""}
EOF
        print_success "Environment file created"
    else
        print_warning ".env file already exists, skipping creation"
    fi
    
    # Source the environment file
    set -a
    source .env
    set +a
    
    print_success "Environment variables loaded"
}

# Setup SSL certificates
setup_ssl() {
    if [[ "$SSL_ENABLED" == "true" ]]; then
        print_status "Setting up SSL certificates..."
        
        mkdir -p nginx/ssl
        
        if [[ "$DEPLOYMENT_ENV" == "production" ]]; then
            print_status "For production, please provide valid SSL certificates"
            print_warning "Place your SSL certificate files in nginx/ssl/ directory:"
            print_warning "  - nginx/ssl/cert.pem"
            print_warning "  - nginx/ssl/key.pem"
        else
            # Generate self-signed certificates for development/staging
            print_status "Generating self-signed SSL certificates for $DEPLOYMENT_ENV..."
            openssl req -x509 -nodes -days 365 -newkey rsa:2048 \\
                -keyout nginx/ssl/key.pem \\
                -out nginx/ssl/cert.pem \\
                -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN"
            print_success "Self-signed SSL certificates generated"
        fi
    fi
}

# Setup configuration files
setup_configs() {
    print_status "Setting up configuration files..."
    
    # Create necessary directories
    mkdir -p monitoring sql nginx config
    
    # PostgreSQL initialization script
    cat > sql/init.sql << 'EOF'
-- NEURICX Database Initialization

-- Create simulation results table
CREATE TABLE IF NOT EXISTS simulation_results (
    id SERIAL PRIMARY KEY,
    economy_id VARCHAR(255) NOT NULL,
    simulation_id VARCHAR(255) UNIQUE NOT NULL,
    n_agents INTEGER NOT NULL,
    n_steps INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(50) DEFAULT 'running',
    results_data JSONB
);

-- Create policy analysis table
CREATE TABLE IF NOT EXISTS policy_analyses (
    id SERIAL PRIMARY KEY,
    economy_id VARCHAR(255) NOT NULL,
    policy_id VARCHAR(255) UNIQUE NOT NULL,
    policy_type VARCHAR(100) NOT NULL,
    policy_parameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    results_data JSONB
);

-- Create crisis predictions table
CREATE TABLE IF NOT EXISTS crisis_predictions (
    id SERIAL PRIMARY KEY,
    economy_id VARCHAR(255) NOT NULL,
    crisis_id VARCHAR(255) UNIQUE NOT NULL,
    prediction_horizon INTEGER,
    crisis_probability DECIMAL(5,4),
    alert_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prediction_data JSONB
);

-- Create risk assessments table
CREATE TABLE IF NOT EXISTS risk_assessments (
    id SERIAL PRIMARY KEY,
    economy_id VARCHAR(255) NOT NULL,
    risk_id VARCHAR(255) UNIQUE NOT NULL,
    aggregate_risk_score DECIMAL(5,4),
    risk_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    assessment_data JSONB
);

-- Create quantum optimizations table
CREATE TABLE IF NOT EXISTS quantum_optimizations (
    id SERIAL PRIMARY KEY,
    economy_id VARCHAR(255) NOT NULL,
    quantum_id VARCHAR(255) UNIQUE NOT NULL,
    optimization_type VARCHAR(100),
    backend VARCHAR(50),
    quantum_advantage BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    optimization_data JSONB
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_simulation_results_economy_id ON simulation_results(economy_id);
CREATE INDEX IF NOT EXISTS idx_policy_analyses_economy_id ON policy_analyses(economy_id);
CREATE INDEX IF NOT EXISTS idx_crisis_predictions_economy_id ON crisis_predictions(economy_id);
CREATE INDEX IF NOT EXISTS idx_risk_assessments_economy_id ON risk_assessments(economy_id);
CREATE INDEX IF NOT EXISTS idx_quantum_optimizations_economy_id ON quantum_optimizations(economy_id);

-- Create monitoring views
CREATE OR REPLACE VIEW simulation_summary AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_simulations,
    AVG(n_agents) as avg_agents,
    AVG(n_steps) as avg_steps,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
    COUNT(CASE WHEN status = 'running' THEN 1 END) as running,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
FROM simulation_results
GROUP BY DATE(created_at)
ORDER BY date DESC;

COMMIT;
EOF

    # Prometheus configuration
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'neuricx-api'
    static_configs:
      - targets: ['neuricx-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'neuricx-streaming'
    static_configs:
      - targets: ['neuricx-streaming:8001']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'neuricx-quantum'
    static_configs:
      - targets: ['neuricx-quantum:8002']
    metrics_path: '/metrics'
    scrape_interval: 60s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF

    # NGINX configuration
    cat > nginx/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server neuricx-api:8000;
    }
    
    upstream dashboard_backend {
        server neuricx-dashboard:3000;
    }
    
    upstream streaming_backend {
        server neuricx-streaming:8001;
    }

    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone \$binary_remote_addr zone=dashboard_limit:10m rate=5r/s;

    server {
        listen 80;
        server_name $DOMAIN;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # API endpoints
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            proxy_pass http://api_backend/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            
            # Increase timeouts for long-running operations
            proxy_connect_timeout 300s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }

        # Dashboard
        location / {
            limit_req zone=dashboard_limit burst=10 nodelay;
            proxy_pass http://dashboard_backend/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        # WebSocket for real-time streaming
        location /ws/ {
            proxy_pass http://streaming_backend/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        # Health check
        location /health {
            access_log off;
            return 200 "healthy\\n";
            add_header Content-Type text/plain;
        }
    }
EOF

    if [[ "$SSL_ENABLED" == "true" ]]; then
        cat >> nginx/nginx.conf << EOF

    server {
        listen 443 ssl http2;
        server_name $DOMAIN;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        # Same location blocks as HTTP server
        # ... (repeat the location blocks from above)
    }
EOF
    fi

    cat >> nginx/nginx.conf << EOF
}
EOF

    # API configuration
    mkdir -p config
    cat > config/api_keys.json << EOF
{
    "alpha_vantage": "$ALPHA_VANTAGE_API_KEY",
    "polygon": "$POLYGON_API_KEY",
    "iex": "",
    "yahoo": ""
}
EOF

    print_success "Configuration files created"
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build all images in parallel where possible
    docker-compose build --parallel
    
    print_success "Docker images built successfully"
}

# Start services
start_services() {
    print_status "Starting NEURICX services..."
    
    # Start infrastructure services first
    docker-compose up -d redis postgres zookeeper kafka
    
    # Wait for infrastructure to be ready
    print_status "Waiting for infrastructure services to be ready..."
    sleep 30
    
    # Start application services
    docker-compose up -d neuricx-api neuricx-streaming neuricx-quantum
    
    # Wait for application services
    sleep 20
    
    # Start frontend and monitoring
    docker-compose up -d neuricx-dashboard nginx prometheus grafana jupyter
    
    print_success "All services started"
}

# Verify deployment
verify_deployment() {
    print_status "Verifying deployment..."
    
    # Check if all containers are running
    print_status "Checking container status..."
    docker-compose ps
    
    # Health checks
    print_status "Performing health checks..."
    
    # API health check
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_success "API service is healthy"
    else
        print_error "API service health check failed"
    fi
    
    # Dashboard check
    if curl -f http://localhost:3000 &> /dev/null; then
        print_success "Dashboard is accessible"
    else
        print_warning "Dashboard may still be starting up"
    fi
    
    # Database check
    if docker-compose exec -T postgres pg_isready -U neuricx &> /dev/null; then
        print_success "Database is ready"
    else
        print_error "Database is not ready"
    fi
    
    # Redis check
    if docker-compose exec -T redis redis-cli ping | grep -q PONG; then
        print_success "Redis is ready"
    else
        print_error "Redis is not ready"
    fi
    
    print_success "Deployment verification completed"
}

# Show deployment summary
show_summary() {
    print_success "NEURICX deployment completed successfully!"
    echo
    echo "================== DEPLOYMENT SUMMARY =================="
    echo "Environment: $DEPLOYMENT_ENV"
    echo "Domain: $DOMAIN"
    echo "SSL Enabled: $SSL_ENABLED"
    echo
    echo "Services are available at:"
    echo "  🌐 Dashboard:     http://$DOMAIN"
    echo "  🔌 API:          http://$DOMAIN/api"
    echo "  📊 Grafana:      http://$DOMAIN:3001 (admin/admin)"
    echo "  📈 Prometheus:   http://$DOMAIN:9090"
    echo "  📓 Jupyter:      http://$DOMAIN:8888 (token: $JUPYTER_TOKEN)"
    echo
    echo "Database:"
    echo "  🗄️  PostgreSQL:   localhost:5432 (neuricx/$POSTGRES_PASSWORD)"
    echo "  🔴 Redis:        localhost:6379"
    echo
    echo "Monitoring:"
    echo "  📈 Prometheus:   http://$DOMAIN:9090"
    echo "  📊 Grafana:      http://$DOMAIN:3001"
    echo
    echo "Useful commands:"
    echo "  📋 View logs:     docker-compose logs -f [service_name]"
    echo "  🔄 Restart:       docker-compose restart [service_name]"
    echo "  🛑 Stop all:      docker-compose down"
    echo "  🗑️  Clean up:      docker-compose down -v --remove-orphans"
    echo "========================================================"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up deployment..."
    docker-compose down -v --remove-orphans
    docker system prune -f
    print_success "Cleanup completed"
}

# Main deployment flow
main() {
    case "${1:-deploy}" in
        "deploy")
            check_dependencies
            setup_environment
            setup_ssl
            setup_configs
            build_images
            start_services
            verify_deployment
            show_summary
            ;;
        "cleanup")
            cleanup
            ;;
        "restart")
            print_status "Restarting NEURICX services..."
            docker-compose restart
            verify_deployment
            ;;
        "logs")
            docker-compose logs -f ${2:-}
            ;;
        "status")
            docker-compose ps
            ;;
        *)
            echo "Usage: $0 {deploy|cleanup|restart|logs|status} [environment] [domain] [ssl_enabled]"
            echo
            echo "Commands:"
            echo "  deploy   - Full deployment (default)"
            echo "  cleanup  - Stop and remove all containers and volumes"
            echo "  restart  - Restart all services"
            echo "  logs     - Show logs (optionally for specific service)"
            echo "  status   - Show container status"
            echo
            echo "Examples:"
            echo "  $0 deploy production example.com true"
            echo "  $0 cleanup"
            echo "  $0 logs neuricx-api"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"