# NEURICX Infrastructure as Code with Terraform
# Supports deployment to AWS, GCP, and Azure

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

# Variables
variable "cloud_provider" {
  description = "Cloud provider (aws, gcp, azure)"
  type        = string
  default     = "aws"
}

variable "environment" {
  description = "Environment (development, staging, production)"
  type        = string
  default     = "development"
}

variable "region" {
  description = "Cloud region"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "neuricx"
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = ""
}

variable "enable_quantum_computing" {
  description = "Enable quantum computing services"
  type        = bool
  default     = false
}

variable "enable_monitoring" {
  description = "Enable monitoring and observability"
  type        = bool
  default     = true
}

variable "enable_ssl" {
  description = "Enable SSL/TLS"
  type        = bool
  default     = true
}

# Local values
locals {
  name_prefix = "${var.project_name}-${var.environment}"
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Component   = "neuricx"
  }
}

# Data sources
data "aws_caller_identity" "current" {
  count = var.cloud_provider == "aws" ? 1 : 0
}

data "aws_region" "current" {
  count = var.cloud_provider == "aws" ? 1 : 0
}

# AWS Provider Configuration
provider "aws" {
  count  = var.cloud_provider == "aws" ? 1 : 0
  region = var.region
  
  default_tags {
    tags = local.common_tags
  }
}

# GCP Provider Configuration
provider "google" {
  count   = var.cloud_provider == "gcp" ? 1 : 0
  project = var.project_name
  region  = var.region
}

# Azure Provider Configuration
provider "azurerm" {
  count = var.cloud_provider == "azure" ? 1 : 0
  features {}
}

# AWS Infrastructure
module "aws_infrastructure" {
  count  = var.cloud_provider == "aws" ? 1 : 0
  source = "./modules/aws"
  
  name_prefix              = local.name_prefix
  environment             = var.environment
  region                  = var.region
  domain_name            = var.domain_name
  enable_quantum_computing = var.enable_quantum_computing
  enable_monitoring       = var.enable_monitoring
  enable_ssl             = var.enable_ssl
  
  tags = local.common_tags
}

# GCP Infrastructure
module "gcp_infrastructure" {
  count  = var.cloud_provider == "gcp" ? 1 : 0
  source = "./modules/gcp"
  
  name_prefix              = local.name_prefix
  environment             = var.environment
  region                  = var.region
  project_id              = var.project_name
  domain_name            = var.domain_name
  enable_quantum_computing = var.enable_quantum_computing
  enable_monitoring       = var.enable_monitoring
  enable_ssl             = var.enable_ssl
}

# Azure Infrastructure
module "azure_infrastructure" {
  count  = var.cloud_provider == "azure" ? 1 : 0
  source = "./modules/azure"
  
  name_prefix              = local.name_prefix
  environment             = var.environment
  location               = var.region
  domain_name            = var.domain_name
  enable_quantum_computing = var.enable_quantum_computing
  enable_monitoring       = var.enable_monitoring
  enable_ssl             = var.enable_ssl
  
  tags = local.common_tags
}

# Kubernetes Provider Configuration
provider "kubernetes" {
  host                   = local.k8s_host
  cluster_ca_certificate = base64decode(local.k8s_cluster_ca_certificate)
  token                  = local.k8s_token
}

provider "helm" {
  kubernetes {
    host                   = local.k8s_host
    cluster_ca_certificate = base64decode(local.k8s_cluster_ca_certificate)
    token                  = local.k8s_token
  }
}

locals {
  # Kubernetes configuration based on cloud provider
  k8s_host = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].cluster_endpoint : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].cluster_endpoint : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].cluster_endpoint : ""
  ) : ""
  
  k8s_cluster_ca_certificate = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].cluster_ca_certificate : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].cluster_ca_certificate : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].cluster_ca_certificate : ""
  ) : ""
  
  k8s_token = var.cloud_provider == "aws" ? (
    length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].cluster_token : ""
  ) : var.cloud_provider == "gcp" ? (
    length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].cluster_token : ""
  ) : var.cloud_provider == "azure" ? (
    length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].cluster_token : ""
  ) : ""
}

# Kubernetes Applications
module "kubernetes_applications" {
  source = "./modules/kubernetes"
  
  name_prefix              = local.name_prefix
  environment             = var.environment
  domain_name            = var.domain_name
  enable_quantum_computing = var.enable_quantum_computing
  enable_monitoring       = var.enable_monitoring
  enable_ssl             = var.enable_ssl
  
  depends_on = [
    module.aws_infrastructure,
    module.gcp_infrastructure,
    module.azure_infrastructure
  ]
}

# Outputs
output "application_url" {
  description = "URL to access the NEURICX application"
  value = var.domain_name != "" ? (
    var.enable_ssl ? "https://${var.domain_name}" : "http://${var.domain_name}"
  ) : (
    var.cloud_provider == "aws" && length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].load_balancer_dns :
    var.cloud_provider == "gcp" && length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].load_balancer_ip :
    var.cloud_provider == "azure" && length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].load_balancer_ip :
    "Not available"
  )
}

output "api_endpoint" {
  description = "API endpoint URL"
  value = "${output.application_url.value}/api"
}

output "dashboard_url" {
  description = "Dashboard URL"
  value = output.application_url.value
}

output "monitoring_urls" {
  description = "Monitoring service URLs"
  value = var.enable_monitoring ? {
    grafana    = "${output.application_url.value}/grafana"
    prometheus = "${output.application_url.value}/prometheus"
  } : {}
}

output "database_connection" {
  description = "Database connection information"
  value = var.cloud_provider == "aws" && length(module.aws_infrastructure) > 0 ? {
    endpoint = module.aws_infrastructure[0].database_endpoint
    port     = module.aws_infrastructure[0].database_port
  } : var.cloud_provider == "gcp" && length(module.gcp_infrastructure) > 0 ? {
    endpoint = module.gcp_infrastructure[0].database_endpoint
    port     = module.gcp_infrastructure[0].database_port
  } : var.cloud_provider == "azure" && length(module.azure_infrastructure) > 0 ? {
    endpoint = module.azure_infrastructure[0].database_endpoint
    port     = module.azure_infrastructure[0].database_port
  } : {}
  sensitive = true
}

output "kubernetes_cluster" {
  description = "Kubernetes cluster information"
  value = {
    endpoint = local.k8s_host
    name = var.cloud_provider == "aws" && length(module.aws_infrastructure) > 0 ? module.aws_infrastructure[0].cluster_name :
           var.cloud_provider == "gcp" && length(module.gcp_infrastructure) > 0 ? module.gcp_infrastructure[0].cluster_name :
           var.cloud_provider == "azure" && length(module.azure_infrastructure) > 0 ? module.azure_infrastructure[0].cluster_name :
           "Not available"
  }
}

output "quantum_services" {
  description = "Quantum computing service information"
  value = var.enable_quantum_computing ? {
    enabled = true
    endpoint = "${output.application_url.value}/quantum"
    simulators = ["statevector", "qasm", "aer"]
    backends = ["ibm", "rigetti", "ionq"]
  } : {
    enabled = false
  }
}