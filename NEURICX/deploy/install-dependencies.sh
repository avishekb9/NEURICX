#!/bin/bash

# NEURICX Dependencies Installation Script
# Installs Docker, Docker Compose, and other required dependencies

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$NAME
            VER=$VERSION_ID
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macOS"
    elif [[ "$OSTYPE" == "msys" ]]; then
        OS="Windows"
    else
        OS="Unknown"
    fi
    
    print_status "Detected OS: $OS"
}

# Install Docker on Ubuntu/Debian
install_docker_ubuntu() {
    print_status "Installing Docker on Ubuntu/Debian..."
    
    # Update package index
    sudo apt-get update
    
    # Install prerequisites
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Set up stable repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Update package index again
    sudo apt-get update
    
    # Install Docker Engine
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    
    print_success "Docker installed successfully!"
    print_warning "Please log out and log back in for group changes to take effect."
}

# Install Docker on macOS
install_docker_macos() {
    print_status "Installing Docker on macOS..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        print_error "Homebrew is not installed. Please install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    # Install Docker using Homebrew
    brew install --cask docker
    
    print_success "Docker installed successfully!"
    print_warning "Please start Docker Desktop from Applications folder."
}

# Install Docker on other Linux distributions
install_docker_linux() {
    print_status "Installing Docker using convenience script..."
    
    # Download and run Docker installation script
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    
    print_success "Docker installed successfully!"
    print_warning "Please log out and log back in for group changes to take effect."
}

# Install Docker Compose (if not included)
install_docker_compose() {
    print_status "Checking Docker Compose..."
    
    if ! docker compose version &> /dev/null; then
        print_status "Installing Docker Compose..."
        
        # Download Docker Compose
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        
        # Make it executable
        sudo chmod +x /usr/local/bin/docker-compose
        
        print_success "Docker Compose installed successfully!"
    else
        print_success "Docker Compose is already installed"
    fi
}

# Install R and required packages
install_r_dependencies() {
    print_status "Installing R and required packages..."
    
    if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
        # Install R
        sudo apt-get update
        sudo apt-get install -y r-base r-base-dev
        
        # Install system dependencies for R packages
        sudo apt-get install -y \
            libcurl4-openssl-dev \
            libssl-dev \
            libxml2-dev \
            libpq-dev \
            libudunits2-dev \
            libgdal-dev \
            libgeos-dev \
            libproj-dev
            
    elif [[ "$OS" == "macOS" ]]; then
        if command -v brew &> /dev/null; then
            brew install r
        else
            print_warning "Please install R from https://cran.r-project.org/"
        fi
    fi
    
    print_success "R dependencies installed!"
}

# Install Python and quantum computing packages
install_python_dependencies() {
    print_status "Installing Python and quantum computing packages..."
    
    # Check if Python 3 is installed
    if ! command -v python3 &> /dev/null; then
        if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
            sudo apt-get install -y python3 python3-pip
        elif [[ "$OS" == "macOS" ]]; then
            if command -v brew &> /dev/null; then
                brew install python3
            fi
        fi
    fi
    
    # Install quantum computing packages
    pip3 install --user qiskit qiskit-aer qiskit-ibm-provider cirq pennylane
    
    print_success "Python quantum packages installed!"
}

# Install Node.js for web interface
install_nodejs() {
    print_status "Installing Node.js..."
    
    if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
        # Install Node.js using NodeSource repository
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
    elif [[ "$OS" == "macOS" ]]; then
        if command -v brew &> /dev/null; then
            brew install node
        fi
    fi
    
    print_success "Node.js installed!"
}

# Verify installations
verify_installations() {
    print_status "Verifying installations..."
    
    # Check Docker
    if docker --version &> /dev/null; then
        print_success "Docker: $(docker --version)"
    else
        print_error "Docker installation failed"
    fi
    
    # Check Docker Compose
    if docker compose version &> /dev/null; then
        print_success "Docker Compose: $(docker compose version)"
    else
        print_error "Docker Compose installation failed"
    fi
    
    # Check R
    if R --version &> /dev/null; then
        print_success "R: $(R --version | head -1)"
    else
        print_warning "R not found - install manually if needed"
    fi
    
    # Check Python
    if python3 --version &> /dev/null; then
        print_success "Python: $(python3 --version)"
    else
        print_warning "Python3 not found"
    fi
    
    # Check Node.js
    if node --version &> /dev/null; then
        print_success "Node.js: $(node --version)"
    else
        print_warning "Node.js not found"
    fi
}

# Main installation function
main() {
    echo "======================================"
    echo "  NEURICX Dependencies Installer"
    echo "======================================"
    echo
    
    detect_os
    
    # Install Docker based on OS
    if ! command -v docker &> /dev/null; then
        case "$OS" in
            *"Ubuntu"*|*"Debian"*)
                install_docker_ubuntu
                ;;
            "macOS")
                install_docker_macos
                ;;
            *)
                install_docker_linux
                ;;
        esac
    else
        print_success "Docker is already installed"
    fi
    
    # Install Docker Compose
    install_docker_compose
    
    # Install other dependencies
    install_r_dependencies
    install_python_dependencies
    install_nodejs
    
    # Verify installations
    verify_installations
    
    echo
    echo "======================================"
    print_success "Installation completed!"
    echo "======================================"
    echo
    print_status "Next steps:"
    echo "1. If you installed Docker for the first time, please log out and log back in"
    echo "2. Start Docker service: sudo systemctl start docker (Linux) or start Docker Desktop (macOS/Windows)"
    echo "3. Run the NEURICX deployment: ./deploy.sh"
    echo
    print_warning "If you see permission errors with Docker, run: newgrp docker"
}

# Run main function
main "$@"