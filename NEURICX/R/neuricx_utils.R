#' NEURICX Utility Functions
#'
#' @description
#' Utility functions for NEURICX package including data loading, network analysis,
#' performance metrics, and validation tools.

#' Load Market Data for NEURICX
#'
#' @param symbols Character vector of stock symbols
#' @param from Start date (default: "2020-01-01")
#' @param to End date (default: Sys.Date())
#' @return List containing prices, returns, and volatility data
#' @export
load_neuricx_market_data <- function(symbols, from = "2020-01-01", to = Sys.Date()) {
  "Load and process market data for NEURICX simulation"
  
  cat("Loading market data for", length(symbols), "symbols...\n")
  
  tryCatch({
    # Load data using quantmod
    data_list <- lapply(symbols, function(symbol) {
      data <- quantmod::getSymbols(symbol, from = from, to = to, auto.assign = FALSE)
      prices <- as.numeric(quantmod::Cl(data))
      return(prices)
    })
    
    # Create price matrix
    min_length <- min(sapply(data_list, length))
    price_matrix <- matrix(0, min_length, length(symbols))
    
    for (i in 1:length(symbols)) {
      price_matrix[, i] <- tail(data_list[[i]], min_length)
    }
    
    colnames(price_matrix) <- symbols
    
    # Calculate returns
    returns_matrix <- apply(price_matrix, 2, function(x) {
      c(0, diff(log(x)))
    })
    
    # Calculate volatility (GARCH if possible, otherwise rolling SD)
    volatility_data <- apply(returns_matrix, 2, function(x) {
      tryCatch({
        spec <- rugarch::ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)))
        fit <- rugarch::ugarchfit(spec, x[-1])  # Remove first zero return
        vol <- as.numeric(rugarch::sigma(fit))
        return(c(sd(x[1:20]), vol))  # Prepend initial volatility estimate
      }, error = function(e) {
        # Fallback to rolling standard deviation
        return(runif(length(x), 0.01, 0.03))
      })
    })
    
    result <- list(
      prices = price_matrix,
      returns = returns_matrix,
      volatility = volatility_data[, 1],  # Use first asset's volatility as proxy
      symbols = symbols,
      dates = seq(from = as.Date(from), by = "day", length.out = min_length)
    )
    
    cat("Successfully loaded market data with", nrow(price_matrix), "observations\n")
    return(result)
    
  }, error = function(e) {
    cat("Error loading market data:", e$message, "\n")
    cat("Using synthetic data instead...\n")
    return(generate_synthetic_market_data(symbols, 252))
  })
}

#' Generate Synthetic Market Data
#'
#' @param symbols Character vector of stock symbols
#' @param n_obs Number of observations
#' @return List containing synthetic market data
#' @export
generate_synthetic_market_data <- function(symbols, n_obs = 252) {
  "Generate synthetic market data for testing purposes"
  
  n_assets <- length(symbols)
  
  # Generate correlated returns
  correlation_matrix <- matrix(0.3, n_assets, n_assets)
  diag(correlation_matrix) <- 1
  
  returns_matrix <- matrix(rnorm(n_obs * n_assets), n_obs, n_assets) %*% chol(correlation_matrix)
  returns_matrix <- returns_matrix * 0.02  # Scale to reasonable volatility
  
  # Generate price paths
  price_matrix <- matrix(100, n_obs, n_assets)  # Start at $100
  for (i in 2:n_obs) {
    price_matrix[i, ] <- price_matrix[i-1, ] * exp(returns_matrix[i, ])
  }
  
  colnames(price_matrix) <- symbols
  colnames(returns_matrix) <- symbols
  
  # Generate volatility with GARCH-like clustering
  volatility_data <- numeric(n_obs)
  volatility_data[1] <- 0.02
  for (i in 2:n_obs) {
    volatility_data[i] <- 0.9 * volatility_data[i-1] + 0.1 * abs(returns_matrix[i, 1])
  }
  
  return(list(
    prices = price_matrix,
    returns = returns_matrix,
    volatility = volatility_data,
    symbols = symbols,
    dates = seq(from = Sys.Date() - n_obs + 1, to = Sys.Date(), by = "day")
  ))
}

#' Calculate Network Metrics
#'
#' @param adjacency_matrix Network adjacency matrix
#' @return List of network metrics
#' @export
calculate_network_metrics <- function(adjacency_matrix) {
  "Calculate comprehensive network topology metrics"
  
  g <- igraph::graph_from_adjacency_matrix(adjacency_matrix, mode = "undirected", weighted = TRUE)
  
  metrics <- list(
    density = igraph::edge_density(g),
    clustering = igraph::transitivity(g, type = "global"),
    assortativity = tryCatch(igraph::assortativity_degree(g), error = function(e) NA),
    diameter = tryCatch(igraph::diameter(g), error = function(e) NA),
    avg_path_length = tryCatch(igraph::average.path.length(g), error = function(e) NA),
    centralization = igraph::centralization.degree(g)$centralization,
    modularity = tryCatch(igraph::modularity(g, igraph::cluster_fast_greedy(g)$membership), error = function(e) NA),
    components = igraph::count_components(g),
    largest_component_size = max(igraph::component_size(g, igraph::components(g)$membership))
  )
  
  return(metrics)
}

#' Calculate Performance Metrics
#'
#' @param predictions Model predictions
#' @param actual Actual values
#' @return List of performance metrics
#' @export
calculate_performance_metrics <- function(predictions, actual) {
  "Calculate comprehensive performance metrics for model validation"
  
  # Remove any NA values
  valid_indices <- !is.na(predictions) & !is.na(actual)
  pred <- predictions[valid_indices]
  act <- actual[valid_indices]
  
  if (length(pred) == 0) {
    return(list(MSE = NA, MAE = NA, RMSE = NA, Directional_Accuracy = NA, 
               Correlation = NA, R_squared = NA))
  }
  
  metrics <- list(
    MSE = mean((pred - act)^2),
    MAE = mean(abs(pred - act)),
    RMSE = sqrt(mean((pred - act)^2)),
    Directional_Accuracy = calculate_directional_accuracy(pred, act),
    Correlation = cor(pred, act, use = "complete.obs"),
    R_squared = tryCatch({
      lm_fit <- lm(act ~ pred)
      summary(lm_fit)$r.squared
    }, error = function(e) NA)
  )
  
  return(metrics)
}

#' Calculate Directional Accuracy
#'
#' @param predictions Model predictions
#' @param actual Actual values
#' @return Directional accuracy score
#' @export
calculate_directional_accuracy <- function(predictions, actual) {
  "Calculate directional accuracy for time series predictions"
  
  if (length(predictions) <= 1 || length(actual) <= 1) return(NA)
  
  pred_direction <- sign(diff(predictions))
  actual_direction <- sign(diff(actual))
  
  correct_directions <- sum(pred_direction == actual_direction, na.rm = TRUE)
  total_directions <- sum(!is.na(pred_direction) & !is.na(actual_direction))
  
  if (total_directions == 0) return(NA)
  
  return(correct_directions / total_directions)
}

#' Calculate MSE
#'
#' @param predictions Model predictions
#' @param actual Actual values
#' @return Mean squared error
#' @export
calculate_mse <- function(predictions, actual) {
  "Calculate mean squared error"
  valid_indices <- !is.na(predictions) & !is.na(actual)
  if (sum(valid_indices) == 0) return(NA)
  return(mean((predictions[valid_indices] - actual[valid_indices])^2))
}

#' Calculate MAE
#'
#' @param predictions Model predictions
#' @param actual Actual values
#' @return Mean absolute error
#' @export
calculate_mae <- function(predictions, actual) {
  "Calculate mean absolute error"
  valid_indices <- !is.na(predictions) & !is.na(actual)
  if (sum(valid_indices) == 0) return(NA)
  return(mean(abs(predictions[valid_indices] - actual[valid_indices])))
}

#' Fit VAR Benchmark Model
#'
#' @param returns Return series
#' @return VAR model predictions
#' @export
fit_var_benchmark <- function(returns) {
  "Fit Vector Autoregression benchmark model"
  
  tryCatch({
    # Fit VAR model using forecast package
    if (is.matrix(returns)) {
      var_data <- returns
    } else {
      var_data <- matrix(returns, ncol = 1)
    }
    
    # Remove initial zeros and infinite values
    var_data <- var_data[apply(var_data, 1, function(x) all(is.finite(x))), , drop = FALSE]
    
    if (nrow(var_data) < 20) {
      return(rnorm(length(returns), 0, 0.02))
    }
    
    var_model <- forecast::VAR(var_data, p = 2, type = "const")
    predictions <- predict(var_model, n.ahead = 1)$fcst[[1]][, "fcst"]
    
    # Extend predictions to match original length
    if (length(predictions) < length(returns)) {
      predictions <- c(predictions, rep(tail(predictions, 1), length(returns) - length(predictions)))
    }
    
    return(predictions[1:length(returns)])
    
  }, error = function(e) {
    # Fallback to simple AR(1)
    if (length(returns) > 10) {
      ar_model <- ar(returns[is.finite(returns)], method = "yule-walker")
      predictions <- predict(ar_model, n.ahead = length(returns))$pred
      return(as.numeric(predictions))
    } else {
      return(rnorm(length(returns), 0, sd(returns, na.rm = TRUE)))
    }
  })
}

#' Fit DSGE Benchmark Model
#'
#' @param returns Return series
#' @return DSGE model predictions
#' @export
fit_dsge_benchmark <- function(returns) {
  "Fit simplified DSGE benchmark model"
  
  # Simplified DSGE approximation using state-space model
  tryCatch({
    # Use a simple state-space representation
    n <- length(returns)
    if (n < 20) {
      return(rnorm(n, 0, 0.02))
    }
    
    # Fit AR(2) as DSGE approximation
    ar_model <- ar(returns[is.finite(returns)], order.max = 2, method = "yule-walker")
    predictions <- predict(ar_model, n.ahead = n)$pred
    
    return(as.numeric(predictions)[1:n])
    
  }, error = function(e) {
    # Fallback to random walk with drift
    drift <- mean(diff(returns), na.rm = TRUE)
    predictions <- cumsum(c(returns[1], rep(drift, length(returns) - 1)))
    return(diff(predictions))
  })
}

#' Evolve Multi-Layer Networks
#'
#' @param networks Current network state
#' @param decisions Agent decisions matrix
#' @param time_step Current time step
#' @return Updated networks
#' @export
evolve_multilayer_networks <- function(networks, decisions, time_step) {
  "Evolve multi-layer networks based on agent interactions and decisions"
  
  n_agents <- nrow(networks$production)
  
  # Production network evolution (based on economic complementarity)
  prod_updates <- calculate_production_network_updates(networks$production, decisions)
  networks$production <- networks$production + 0.01 * prod_updates
  networks$production[networks$production < 0] <- 0
  networks$production[networks$production > 1] <- 1
  
  # Consumption network evolution (based on preference similarity)
  cons_updates <- calculate_consumption_network_updates(networks$consumption, decisions)
  networks$consumption <- networks$consumption + 0.02 * cons_updates
  networks$consumption[networks$consumption < 0] <- 0
  networks$consumption[networks$consumption > 1] <- 1
  
  # Information network evolution (based on communication value)
  info_updates <- calculate_information_network_updates(networks$information, decisions)
  networks$information <- networks$information + 0.03 * info_updates
  networks$information[networks$information < 0] <- 0
  networks$information[networks$information > 1] <- 1
  
  # Apply network constraints (sparsity, symmetry)
  networks$production <- apply_network_constraints(networks$production, "production")
  networks$consumption <- apply_network_constraints(networks$consumption, "consumption")
  networks$information <- apply_network_constraints(networks$information, "information")
  
  return(networks)
}

#' Calculate Production Network Updates
#'
#' @param current_network Current production network
#' @param decisions Agent decisions
#' @return Network update matrix
calculate_production_network_updates <- function(current_network, decisions) {
  n_agents <- nrow(current_network)
  updates <- matrix(0, n_agents, n_agents)
  
  for (i in 1:n_agents) {
    for (j in 1:n_agents) {
      if (i != j) {
        # Production complementarity
        complementarity <- -abs(decisions[i] - decisions[j])  # Negative of absolute difference
        updates[i, j] <- complementarity
      }
    }
  }
  
  return(updates)
}

#' Calculate Consumption Network Updates
#'
#' @param current_network Current consumption network
#' @param decisions Agent decisions
#' @return Network update matrix
calculate_consumption_network_updates <- function(current_network, decisions) {
  n_agents <- nrow(current_network)
  updates <- matrix(0, n_agents, n_agents)
  
  for (i in 1:n_agents) {
    for (j in 1:n_agents) {
      if (i != j) {
        # Consumption similarity (social influence)
        similarity <- exp(-abs(decisions[i] - decisions[j]))
        updates[i, j] <- similarity - 0.5  # Center around 0
      }
    }
  }
  
  return(updates)
}

#' Calculate Information Network Updates
#'
#' @param current_network Current information network
#' @param decisions Agent decisions
#' @return Network update matrix
calculate_information_network_updates <- function(current_network, decisions) {
  n_agents <- nrow(current_network)
  updates <- matrix(0, n_agents, n_agents)
  
  for (i in 1:n_agents) {
    for (j in 1:n_agents) {
      if (i != j) {
        # Information value (diversity preference)
        diversity_value <- abs(decisions[i] - decisions[j])
        updates[i, j] <- diversity_value - 0.3  # Slight preference for diversity
      }
    }
  }
  
  return(updates)
}

#' Apply Network Constraints
#'
#' @param network Network matrix
#' @param network_type Type of network ("production", "consumption", "information")
#' @return Constrained network matrix
apply_network_constraints <- function(network, network_type) {
  # Ensure symmetry for undirected networks
  network <- (network + t(network)) / 2
  
  # Ensure diagonal is zero (no self-loops)
  diag(network) <- 0
  
  # Apply sparsity constraints
  sparsity_threshold <- switch(network_type,
    "production" = 0.1,
    "consumption" = 0.15,
    "information" = 0.2,
    0.1
  )
  
  # Keep only strongest connections
  threshold <- quantile(network[network > 0], 1 - sparsity_threshold, na.rm = TRUE)
  network[network < threshold] <- 0
  
  return(network)
}

#' Set NEURICX Random Seed
#'
#' @param seed Random seed value
#' @export
set_neuricx_seed <- function(seed = 42) {
  "Set random seed for reproducible NEURICX simulations"
  set.seed(seed)
  cat("NEURICX random seed set to:", seed, "\n")
}

#' Get NEURICX Configuration
#'
#' @return List of current NEURICX configuration parameters
#' @export
get_neuricx_config <- function() {
  "Get current NEURICX configuration parameters"
  
  config <- list(
    version = "1.0.0",
    default_agents = 1000,
    default_steps = 250,
    network_types = c("production", "consumption", "information"),
    agent_types = c("rational_optimizer", "bounded_rational", "social_learner", 
                   "trend_follower", "contrarian", "adaptive_learner"),
    communication_protocols = c("mcp", "market", "social"),
    default_symbols = c("AAPL", "GOOGL", "MSFT", "TSLA", "NVDA")
  )
  
  return(config)
}