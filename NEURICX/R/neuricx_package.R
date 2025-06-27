#' NEURICX: Network-Enhanced Unified Rational Intelligence in Computational Economics
#'
#' @description
#' A comprehensive framework for network-enhanced economic modeling that integrates 
#' heterogeneous agent-based systems with multi-protocol communication. NEURICX 
#' provides tractable implementations of production networks, consumption networks, 
#' and information networks with empirical validation capabilities.
#'
#' @details
#' The NEURICX package implements the complete mathematical framework described in
#' "NEURICX: A Tractable Framework for Network-Enhanced Unified Rational Intelligence
#' in Computational Economics". The package provides:
#'
#' \strong{Core Components:}
#' \itemize{
#'   \item Heterogeneous agent-based modeling with six distinct agent types
#'   \item Multi-layer network dynamics (production, consumption, information)
#'   \item Multi-protocol communication systems (MCP, market, social)
#'   \item Collective intelligence emergence tracking
#'   \item Comprehensive empirical validation framework
#' }
#'
#' \strong{Agent Types:}
#' \itemize{
#'   \item \code{rational_optimizer}: Full optimization with perfect information
#'   \item \code{bounded_rational}: Simplified heuristics with computational limits
#'   \item \code{social_learner}: Imitation and peer learning
#'   \item \code{trend_follower}: Momentum-based strategies
#'   \item \code{contrarian}: Counter-trend strategies
#'   \item \code{adaptive_learner}: Reinforcement learning adaptation
#' }
#'
#' \strong{Network Layers:}
#' \itemize{
#'   \item \code{Production Networks}: Economic complementarity-based connections
#'   \item \code{Consumption Networks}: Social influence and preference similarity
#'   \item \code{Information Networks}: Communication value and diversity preference
#' }
#'
#' \strong{Key Functions:}
#' \itemize{
#'   \item \code{\link{run_neuricx_analysis}()}: Execute complete analysis
#'   \item \code{\link{create_neuricx_economy}()}: Initialize economic system
#'   \item \code{\link{create_neuricx_agent}()}: Create individual agents
#'   \item \code{\link{generate_neuricx_report}()}: Generate visualization report
#'   \item \code{\link{validate_neuricx_framework}()}: Comprehensive validation
#' }
#'
#' @section Getting Started:
#' To run a basic NEURICX analysis:
#'
#' \preformatted{
#' library(NEURICX)
#' 
#' # Set random seed for reproducibility
#' set_neuricx_seed(42)
#' 
#' # Run comprehensive analysis
#' results <- run_neuricx_analysis(
#'   symbols = c("AAPL", "MSFT", "GOOGL"),
#'   n_agents = 1000,
#'   n_steps = 250
#' )
#' 
#' # Generate summary report
#' summary <- create_neuricx_summary_report(results)
#' 
#' # Create visualizations
#' plots <- generate_neuricx_report(results)
#' }
#'
#' @section Advanced Usage:
#' For more control over the simulation:
#'
#' \preformatted{
#' # Create custom economy
#' economy <- create_neuricx_economy(
#'   n_agents = 500,
#'   symbols = c("AAPL", "TSLA"),
#'   network_density = 0.08
#' )
#' 
#' # Run simulation
#' sim_results <- economy$run_simulation(n_steps = 100)
#' 
#' # Analyze collective intelligence
#' ci_analysis <- economy$analyze_collective_intelligence()
#' 
#' # Compare with benchmarks
#' benchmarks <- economy$compare_with_benchmarks()
#' 
#' # Validate framework
#' validation <- economy$validate_framework()
#' }
#'
#' @section Network Analysis:
#' Analyze network evolution patterns:
#'
#' \preformatted{
#' # Analyze network structure evolution
#' network_analysis <- analyze_network_structure(results$simulation_results$network_evolution)
#' 
#' # Plot network evolution
#' network_plots <- plot_network_evolution(network_analysis)
#' 
#' # Create interactive network visualization
#' interactive_net <- create_interactive_network(
#'   results$simulation_results$network_evolution$information[250, , ],
#'   results$simulation_results$agent_types
#' )
#' }
#'
#' @section Data Requirements:
#' NEURICX can work with:
#' \itemize{
#'   \item Real market data (automatically downloaded via quantmod)
#'   \item Synthetic data (generated internally for testing)
#'   \item Custom data (user-provided price/return series)
#' }
#'
#' @section Performance Considerations:
#' \itemize{
#'   \item Computational complexity: O(N²) per time step where N = number of agents
#'   \item Memory usage scales linearly with agents and time steps
#'   \item Recommended: N ≤ 1000 agents for standard analysis
#'   \item For large-scale analysis: Use sparse network approximations
#' }
#'
#' @section Validation Framework:
#' NEURICX includes comprehensive validation:
#' \itemize{
#'   \item Agent behavior validation (type-specific pattern verification)
#'   \item Network formation validation (theoretical consistency)
#'   \item Collective intelligence validation (emergence detection)
#'   \item Economic realism validation (stylized facts reproduction)
#'   \item Statistical validation (distributional properties)
#'   \item Robustness validation (sensitivity analysis)
#' }
#'
#' @section Empirical Applications:
#' NEURICX enables analysis of:
#' \itemize{
#'   \item Monetary policy transmission through networks
#'   \item Financial contagion and systemic risk
#'   \item Technology adoption and diffusion
#'   \item Market microstructure and liquidity
#'   \item Income inequality and wealth dynamics
#'   \item Crisis prediction and intervention
#' }
#'
#' @references
#' \itemize{
#'   \item Acemoglu, D., Carvalho, V. M., Ozdaglar, A., & Tahbaz-Salehi, A. (2012). 
#'         The network origins of aggregate fluctuations. American Economic Review, 102(1), 131-166.
#'   \item Axtell, R., & Farmer, D. (2025). Agent-Based Modeling in Economics and Finance: 
#'         Past, Present, and Future. Journal of Economic Literature, 63(1), 197-287.
#'   \item NEURICX Consortium (2025). NEURICX: A Tractable Framework for Network-Enhanced 
#'         Unified Rational Intelligence in Computational Economics. Working Paper.
#' }
#'
#' @author NEURICX Consortium
#' @keywords economic-modeling agent-based-models network-economics collective-intelligence
#'
#' @docType package
#' @name NEURICX-package
#' @aliases NEURICX
NULL

#' @importFrom methods setRefClass
#' @importFrom stats rnorm runif sd var cor lm
#' @importFrom utils head tail
#' @import igraph
#' @import ggplot2
#' @import dplyr
NULL

# Global variables to avoid R CMD check notes
globalVariables(c(
  "agent_type", "total_return", "avg_sharpe_ratio", "avg_volatility", 
  "avg_total_return", "count", "step", "density", "layer", "clustering",
  "centralization", "gini", "wealth", "agent_id", "ci_index", "emergence_threshold",
  "ci", "MSE", "MAE", "Directional_Accuracy", "Model", "avg_wealth", "value",
  "metric", "time_point", "weight", "x_from", "y_from", "x_to", "y_to",
  "x", "y", "from", "to", "name", "group", ".", "avg_density", "avg_clustering",
  "avg_centralization"
))

#' NEURICX Configuration
#'
#' @description
#' Global configuration parameters for NEURICX package
#'
#' @format A list containing:
#' \describe{
#'   \item{version}{Package version}
#'   \item{default_agents}{Default number of agents}
#'   \item{default_steps}{Default simulation steps}
#'   \item{network_types}{Available network types}
#'   \item{agent_types}{Available agent types}
#'   \item{communication_protocols}{Available communication protocols}
#' }
#' @export
NEURICX_CONFIG <- list(
  version = "1.0.0",
  default_agents = 1000,
  default_steps = 250,
  network_types = c("production", "consumption", "information"),
  agent_types = c("rational_optimizer", "bounded_rational", "social_learner", 
                 "trend_follower", "contrarian", "adaptive_learner"),
  communication_protocols = c("mcp", "market", "social"),
  default_symbols = c("AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"),
  validation_thresholds = list(
    emergence_threshold = 1.2,
    network_stability = 0.8,
    agent_consistency = 0.7,
    overall_validation = 0.75
  )
)

.onAttach <- function(libname, pkgname) {
  packageStartupMessage(
    paste0("NEURICX v", NEURICX_CONFIG$version, ": Network-Enhanced Economic Modeling\n"),
    "Type '?NEURICX' for help, 'demo(neuricx)' for examples.\n",
    "Visit https://github.com/neuricx-economics/NEURICX for documentation."
  )
}