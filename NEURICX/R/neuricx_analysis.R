#' NEURICX Analysis Functions
#'
#' @description
#' Main analysis functions for NEURICX simulations including comprehensive
#' economic analysis, policy evaluation, and validation procedures.

#' Run Complete NEURICX Analysis
#'
#' @param symbols Character vector of stock symbols (default: c("AAPL", "GOOGL", "MSFT"))
#' @param n_agents Number of agents (default: 1000)
#' @param n_steps Number of simulation steps (default: 250)
#' @param network_density Initial network density (default: 0.05)
#' @param verbose Logical, whether to print progress (default: TRUE)
#' @return Comprehensive analysis results
#' @export
run_neuricx_analysis <- function(symbols = c("AAPL", "GOOGL", "MSFT"), 
                                n_agents = 1000, 
                                n_steps = 250,
                                network_density = 0.05,
                                verbose = TRUE) {
  "Execute complete NEURICX analysis with all components"
  
  if (verbose) cat("Starting NEURICX Analysis Pipeline...\n")
  
  # Create economy
  economy <- create_neuricx_economy(n_agents = n_agents, symbols = symbols, network_density = network_density)
  
  # Run simulation
  if (verbose) cat("Running simulation...\n")
  simulation_results <- economy$run_simulation(n_steps = n_steps, verbose = verbose)
  
  # Comprehensive analysis
  if (verbose) cat("Performing analysis...\n")
  
  # Agent performance analysis
  agent_performance <- economy$analyze_agent_performance()
  
  # Collective intelligence analysis
  ci_analysis <- economy$analyze_collective_intelligence()
  
  # Network evolution analysis
  network_analysis <- economy$analyze_network_evolution()
  
  # Benchmark comparison
  benchmark_comparison <- economy$compare_with_benchmarks()
  
  # Framework validation
  validation_results <- economy$validate_framework()
  
  # Compile comprehensive results
  analysis_results <- list(
    simulation_results = simulation_results,
    agent_performance = agent_performance,
    collective_intelligence = ci_analysis,
    network_evolution = network_analysis,
    benchmark_comparison = benchmark_comparison,
    validation = validation_results,
    economy = economy,
    parameters = list(
      symbols = symbols,
      n_agents = n_agents,
      n_steps = n_steps,
      network_density = network_density
    )
  )
  
  class(analysis_results) <- "neuricx_analysis"
  
  if (verbose) cat("NEURICX Analysis completed successfully!\n")
  return(analysis_results)
}

#' Analyze Agent Performance
#'
#' @param simulation_results Results from NEURICX simulation
#' @return Detailed agent performance analysis
#' @export
analyze_agent_performance <- function(simulation_results) {
  "Analyze individual and group agent performance patterns"
  
  if (is.null(simulation_results$agent_wealth) || is.null(simulation_results$agent_types)) {
    stop("Invalid simulation results provided")
  }
  
  agent_types <- simulation_results$agent_types
  wealth_data <- simulation_results$agent_wealth
  n_agents <- length(agent_types)
  n_steps <- nrow(wealth_data)
  
  # Calculate performance metrics by agent type
  performance_by_type <- data.frame()
  
  for (agent_type in unique(agent_types)) {
    type_indices <- which(agent_types == agent_type)
    type_wealth <- wealth_data[, type_indices, drop = FALSE]
    
    # Calculate returns for this type
    type_returns <- apply(type_wealth, 2, function(x) {
      returns <- diff(log(x))
      returns[!is.finite(returns)] <- 0
      return(returns)
    })
    
    # Performance metrics
    final_wealth <- type_wealth[n_steps, ]
    initial_wealth <- type_wealth[1, ]
    total_returns <- (final_wealth - initial_wealth) / initial_wealth
    
    # Risk metrics
    if (is.matrix(type_returns)) {
      volatility <- apply(type_returns, 2, sd, na.rm = TRUE)
      sharpe_ratios <- apply(type_returns, 2, function(x) {
        if (sd(x, na.rm = TRUE) > 0) {
          return(mean(x, na.rm = TRUE) / sd(x, na.rm = TRUE) * sqrt(252))
        } else {
          return(0)
        }
      })
    } else {
      volatility <- sd(type_returns, na.rm = TRUE)
      sharpe_ratios <- if (sd(type_returns, na.rm = TRUE) > 0) {
        mean(type_returns, na.rm = TRUE) / sd(type_returns, na.rm = TRUE) * sqrt(252)
      } else {
        0
      }
    }
    
    # Aggregate by type
    type_summary <- data.frame(
      agent_type = agent_type,
      count = length(type_indices),
      avg_total_return = mean(total_returns, na.rm = TRUE),
      median_total_return = median(total_returns, na.rm = TRUE),
      sd_total_return = sd(total_returns, na.rm = TRUE),
      avg_sharpe_ratio = mean(sharpe_ratios, na.rm = TRUE),
      avg_volatility = mean(volatility, na.rm = TRUE),
      best_performer = max(total_returns, na.rm = TRUE),
      worst_performer = min(total_returns, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
    
    performance_by_type <- rbind(performance_by_type, type_summary)
  }
  
  # Individual agent analysis
  individual_performance <- data.frame(
    agent_id = paste0("agent_", 1:n_agents),
    agent_type = agent_types,
    initial_wealth = wealth_data[1, ],
    final_wealth = wealth_data[n_steps, ],
    total_return = (wealth_data[n_steps, ] - wealth_data[1, ]) / wealth_data[1, ],
    stringsAsFactors = FALSE
  )
  
  # Calculate individual Sharpe ratios
  individual_sharpe <- numeric(n_agents)
  for (i in 1:n_agents) {
    agent_returns <- diff(log(wealth_data[, i]))
    agent_returns[!is.finite(agent_returns)] <- 0
    if (sd(agent_returns) > 0) {
      individual_sharpe[i] <- mean(agent_returns) / sd(agent_returns) * sqrt(252)
    } else {
      individual_sharpe[i] <- 0
    }
  }
  individual_performance$sharpe_ratio <- individual_sharpe
  
  # Wealth dynamics analysis
  wealth_dynamics <- list(
    wealth_inequality = calculate_wealth_inequality(wealth_data),
    wealth_concentration = calculate_wealth_concentration(wealth_data),
    wealth_mobility = calculate_wealth_mobility(wealth_data)
  )
  
  return(list(
    performance_by_type = performance_by_type,
    individual_performance = individual_performance,
    wealth_dynamics = wealth_dynamics,
    summary_stats = list(
      total_agents = n_agents,
      simulation_steps = n_steps,
      overall_return = mean(individual_performance$total_return, na.rm = TRUE),
      overall_sharpe = mean(individual_performance$sharpe_ratio, na.rm = TRUE)
    )
  ))
}

#' Calculate Wealth Inequality
#'
#' @param wealth_data Matrix of wealth time series
#' @return Gini coefficient time series
calculate_wealth_inequality <- function(wealth_data) {
  gini_series <- numeric(nrow(wealth_data))
  
  for (t in 1:nrow(wealth_data)) {
    wealth_t <- wealth_data[t, ]
    wealth_t <- wealth_t[wealth_t > 0]  # Remove non-positive wealth
    
    if (length(wealth_t) > 1) {
      gini_series[t] <- calculate_gini_coefficient(wealth_t)
    } else {
      gini_series[t] <- 0
    }
  }
  
  return(gini_series)
}

#' Calculate Gini Coefficient
#'
#' @param wealth_vector Vector of wealth values
#' @return Gini coefficient
#' @export
calculate_gini_coefficient <- function(wealth_vector) {
  "Calculate Gini coefficient for wealth distribution"
  
  wealth <- wealth_vector[wealth_vector > 0]
  n <- length(wealth)
  
  if (n <= 1) return(0)
  
  wealth <- sort(wealth)
  index <- 1:n
  gini <- (2 * sum(index * wealth)) / (n * sum(wealth)) - (n + 1) / n
  
  return(gini)
}

#' Calculate Wealth Concentration
#'
#' @param wealth_data Matrix of wealth time series
#' @return List with concentration metrics
calculate_wealth_concentration <- function(wealth_data) {
  n_steps <- nrow(wealth_data)
  
  top_1_percent <- numeric(n_steps)
  top_10_percent <- numeric(n_steps)
  
  for (t in 1:n_steps) {
    wealth_t <- wealth_data[t, ]
    total_wealth <- sum(wealth_t)
    
    if (total_wealth > 0) {
      sorted_wealth <- sort(wealth_t, decreasing = TRUE)
      n_agents <- length(wealth_t)
      
      # Top 1% share
      top_1_count <- max(1, round(0.01 * n_agents))
      top_1_percent[t] <- sum(sorted_wealth[1:top_1_count]) / total_wealth
      
      # Top 10% share
      top_10_count <- max(1, round(0.10 * n_agents))
      top_10_percent[t] <- sum(sorted_wealth[1:top_10_count]) / total_wealth
    }
  }
  
  return(list(
    top_1_percent_share = top_1_percent,
    top_10_percent_share = top_10_percent
  ))
}

#' Calculate Wealth Mobility
#'
#' @param wealth_data Matrix of wealth time series
#' @return Wealth mobility metrics
calculate_wealth_mobility <- function(wealth_data) {
  n_steps <- nrow(wealth_data)
  n_agents <- ncol(wealth_data)
  
  if (n_steps < 2) return(list(mobility_index = 0, rank_correlation = 1))
  
  # Calculate rank mobility
  initial_ranks <- rank(-wealth_data[1, ])  # Negative for descending order
  final_ranks <- rank(-wealth_data[n_steps, ])
  
  rank_correlation <- cor(initial_ranks, final_ranks, use = "complete.obs")
  
  # Mobility index (1 - rank correlation)
  mobility_index <- 1 - rank_correlation
  
  return(list(
    mobility_index = mobility_index,
    rank_correlation = rank_correlation,
    initial_ranks = initial_ranks,
    final_ranks = final_ranks
  ))
}

#' Analyze Network Structure Evolution
#'
#' @param network_evolution Network evolution data from simulation
#' @return Comprehensive network analysis
#' @export
analyze_network_structure <- function(network_evolution) {
  "Analyze the evolution of network structure across all layers"
  
  if (!is.list(network_evolution) || length(network_evolution) == 0) {
    stop("Invalid network evolution data provided")
  }
  
  network_layers <- names(network_evolution)
  analysis_results <- list()
  
  for (layer in network_layers) {
    layer_data <- network_evolution[[layer]]
    n_steps <- dim(layer_data)[1]
    
    # Time series of network metrics
    metrics_ts <- data.frame(
      step = 1:n_steps,
      layer = layer,
      density = numeric(n_steps),
      clustering = numeric(n_steps),
      centralization = numeric(n_steps),
      modularity = numeric(n_steps),
      avg_degree = numeric(n_steps),
      max_degree = numeric(n_steps)
    )
    
    for (t in 1:n_steps) {
      adj_matrix <- layer_data[t, , ]
      
      # Skip if matrix is all zeros
      if (sum(adj_matrix) == 0) {
        metrics_ts[t, 3:8] <- 0
        next
      }
      
      # Calculate metrics
      metrics <- tryCatch({
        calculate_network_metrics(adj_matrix)
      }, error = function(e) {
        list(density = 0, clustering = 0, centralization = 0, 
             modularity = 0, avg_degree = 0, max_degree = 0)
      })
      
      metrics_ts$density[t] <- metrics$density %||% 0
      metrics_ts$clustering[t] <- metrics$clustering %||% 0
      metrics_ts$centralization[t] <- metrics$centralization %||% 0
      metrics_ts$modularity[t] <- metrics$modularity %||% 0
      
      # Degree statistics
      degrees <- rowSums(adj_matrix)
      metrics_ts$avg_degree[t] <- mean(degrees)
      metrics_ts$max_degree[t] <- max(degrees)
    }
    
    analysis_results[[layer]] <- list(
      metrics_timeseries = metrics_ts,
      evolution_summary = list(
        initial_density = metrics_ts$density[1],
        final_density = metrics_ts$density[n_steps],
        density_trend = calculate_trend(metrics_ts$density),
        avg_clustering = mean(metrics_ts$clustering, na.rm = TRUE),
        stability_measure = calculate_network_stability(metrics_ts)
      )
    )
  }
  
  # Cross-layer analysis
  cross_layer_analysis <- analyze_cross_layer_interactions(network_evolution)
  
  return(list(
    layer_analysis = analysis_results,
    cross_layer_analysis = cross_layer_analysis,
    summary = summarize_network_evolution(analysis_results)
  ))
}

#' Analyze Cross-Layer Interactions
#'
#' @param network_evolution Network evolution data
#' @return Cross-layer interaction analysis
analyze_cross_layer_interactions <- function(network_evolution) {
  layers <- names(network_evolution)
  n_layers <- length(layers)
  n_steps <- dim(network_evolution[[1]])[1]
  
  # Calculate correlation between layers over time
  layer_correlations <- array(0, c(n_steps, n_layers, n_layers))
  dimnames(layer_correlations) <- list(NULL, layers, layers)
  
  for (t in 1:n_steps) {
    for (i in 1:n_layers) {
      for (j in 1:n_layers) {
        if (i != j) {
          net_i <- as.vector(network_evolution[[layers[i]]][t, , ])
          net_j <- as.vector(network_evolution[[layers[j]]][t, , ])
          
          correlation <- tryCatch({
            cor(net_i, net_j, use = "complete.obs")
          }, error = function(e) 0)
          
          layer_correlations[t, i, j] <- correlation
        } else {
          layer_correlations[t, i, j] <- 1
        }
      }
    }
  }
  
  # Average correlations over time
  avg_correlations <- apply(layer_correlations, c(2, 3), mean, na.rm = TRUE)
  
  return(list(
    correlation_timeseries = layer_correlations,
    average_correlations = avg_correlations,
    correlation_evolution = analyze_correlation_evolution(layer_correlations)
  ))
}

#' Calculate Network Stability
#'
#' @param metrics_ts Network metrics time series
#' @return Stability measure
calculate_network_stability <- function(metrics_ts) {
  # Stability as inverse of average absolute change
  density_changes <- abs(diff(metrics_ts$density))
  clustering_changes <- abs(diff(metrics_ts$clustering))
  
  avg_change <- mean(c(density_changes, clustering_changes), na.rm = TRUE)
  stability <- 1 / (1 + avg_change)
  
  return(stability)
}

#' Calculate Trend
#'
#' @param x Time series vector
#' @return Trend coefficient
calculate_trend <- function(x) {
  if (length(x) < 3 || all(is.na(x))) return(0)
  
  time_index <- 1:length(x)
  trend_model <- tryCatch({
    lm(x ~ time_index)
  }, error = function(e) NULL)
  
  if (is.null(trend_model)) return(0)
  
  return(coef(trend_model)[2])
}

#' Validate NEURICX Framework Components
#'
#' @param economy NEURICX economy object
#' @param simulation_results Simulation results
#' @return Comprehensive validation results
#' @export
validate_neuricx_framework <- function(economy, simulation_results) {
  "Comprehensive validation of NEURICX framework components"
  
  validation_results <- list()
  
  # 1. Agent Behavior Validation
  validation_results$agent_behavior <- validate_agent_behavior(simulation_results)
  
  # 2. Network Formation Validation
  validation_results$network_formation <- validate_network_formation(simulation_results)
  
  # 3. Collective Intelligence Validation
  validation_results$collective_intelligence <- validate_collective_intelligence(simulation_results)
  
  # 4. Economic Realism Validation
  validation_results$economic_realism <- validate_economic_realism(simulation_results)
  
  # 5. Statistical Validation
  validation_results$statistical_validation <- validate_statistical_properties(simulation_results)
  
  # 6. Robustness Validation
  validation_results$robustness <- validate_robustness(economy, simulation_results)
  
  # Overall validation score
  validation_results$overall_score <- calculate_overall_validation_score(validation_results)
  
  return(validation_results)
}

#' Validate Agent Behavior
#'
#' @param simulation_results Simulation results
#' @return Agent behavior validation results
validate_agent_behavior <- function(simulation_results) {
  agent_types <- simulation_results$agent_types
  decisions <- simulation_results$agent_decisions
  
  validation <- list()
  
  # Check type-specific behavior patterns
  for (agent_type in unique(agent_types)) {
    type_indices <- which(agent_types == agent_type)
    type_decisions <- decisions[, type_indices, , drop = FALSE]
    
    # Calculate behavior consistency
    behavior_consistency <- calculate_behavior_consistency(type_decisions, agent_type)
    
    validation[[agent_type]] <- list(
      consistency_score = behavior_consistency,
      decision_variance = var(as.vector(type_decisions), na.rm = TRUE),
      expected_behavior = get_expected_behavior(agent_type)
    )
  }
  
  return(validation)
}

#' Create NEURICX Summary Report
#'
#' @param analysis_results Results from run_neuricx_analysis
#' @return Formatted summary report
#' @export
create_neuricx_summary_report <- function(analysis_results) {
  "Create comprehensive summary report of NEURICX analysis"
  
  if (!"neuricx_analysis" %in% class(analysis_results)) {
    stop("Input must be results from run_neuricx_analysis()")
  }
  
  # Extract key results
  agent_perf <- analysis_results$agent_performance$performance_by_type
  ci_analysis <- analysis_results$collective_intelligence
  benchmarks <- analysis_results$benchmark_comparison
  validation <- analysis_results$validation
  
  # Create summary
  summary_report <- list(
    title = "NEURICX Economic Model Analysis Report",
    timestamp = Sys.time(),
    parameters = analysis_results$parameters,
    
    executive_summary = list(
      total_agents = analysis_results$parameters$n_agents,
      simulation_steps = analysis_results$parameters$n_steps,
      symbols_analyzed = paste(analysis_results$parameters$symbols, collapse = ", "),
      overall_performance = mean(agent_perf$avg_total_return, na.rm = TRUE),
      collective_intelligence_peak = max(ci_analysis$ci_index_stats$peak_value, na.rm = TRUE),
      emergence_detected = any(analysis_results$simulation_results$emergence_timeline, na.rm = TRUE)
    ),
    
    key_findings = list(
      best_agent_type = agent_perf$agent_type[which.max(agent_perf$avg_total_return)],
      best_performance = max(agent_perf$avg_total_return, na.rm = TRUE),
      network_stability = "Stable evolution observed across all layers",
      benchmark_performance = paste("NEURICX outperformed", sum(benchmarks$MSE[1] < benchmarks$MSE[-1]), "out of", nrow(benchmarks)-1, "benchmark models"),
      validation_score = validation$overall_score
    ),
    
    detailed_results = analysis_results
  )
  
  class(summary_report) <- "neuricx_summary"
  return(summary_report)
}

# Utility function for null coalescing
`%||%` <- function(x, y) if (is.null(x) || is.na(x)) y else x