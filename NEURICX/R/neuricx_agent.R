#' NEURICX Agent Class
#'
#' @description
#' Implementation of heterogeneous agents with multi-protocol communication
#' capabilities for the NEURICX framework. Each agent has network positions
#' across production, consumption, and information networks.
#'
#' @details
#' The NEURICXAgent class implements six agent types:
#' \itemize{
#'   \item rational_optimizer: Full optimization with perfect information
#'   \item bounded_rational: Simplified heuristics with computational limits
#'   \item social_learner: Imitation and peer learning
#'   \item trend_follower: Momentum-based strategies
#'   \item contrarian: Counter-trend strategies
#'   \item adaptive_learner: Reinforcement learning adaptation
#' }
#'
#' @field id Character string identifying the agent
#' @field type Agent type (one of six types)
#' @field wealth Current wealth of the agent
#' @field state Individual state vector
#' @field networks List of network positions (production, consumption, information)
#' @field communication Communication protocol preferences and costs
#' @field learning Learning parameters and history
#'
#' @examples
#' \dontrun{
#' # Create a rational optimizer agent
#' agent <- NEURICXAgent$new("agent_001", "rational_optimizer", 1000000)
#' 
#' # Make a decision based on network information
#' decision <- agent$make_decision(market_data, network_signals)
#' 
#' # Update wealth and learning
#' agent$update_wealth(0.02)
#' }
#'
#' @export
NEURICXAgent <- setRefClass("NEURICXAgent",
  fields = list(
    id = "character",
    type = "character", 
    wealth = "numeric",
    state = "list",
    networks = "list",
    communication = "list",
    learning = "list"
  ),
  
  methods = list(
    #' Initialize Agent
    #' 
    #' @param agent_id Character string for agent ID
    #' @param agent_type Agent type (rational_optimizer, bounded_rational, etc.)
    #' @param initial_wealth Initial wealth amount (default: 1,000,000)
    initialize = function(agent_id, agent_type, initial_wealth = 1000000) {
      "Initialize a new NEURICX agent with specified parameters"
      
      id <<- agent_id
      type <<- agent_type
      wealth <<- initial_wealth
      
      # Initialize state vector
      state <<- list(
        preferences = runif(10, 0, 1),
        risk_tolerance = runif(1, 0.1, 0.9),
        information_capacity = switch(type,
          "rational_optimizer" = 1.0,
          "bounded_rational" = 0.7,
          "social_learner" = 0.6,
          "trend_follower" = 0.5,
          "contrarian" = 0.5,
          "adaptive_learner" = 0.8,
          0.6
        ),
        memory_horizon = switch(type,
          "rational_optimizer" = 100,
          "bounded_rational" = 20,
          "social_learner" = 10,
          "trend_follower" = 15,
          "contrarian" = 15,
          "adaptive_learner" = 50,
          20
        )
      )
      
      # Initialize network positions
      networks <<- list(
        production = list(connections = numeric(0), weights = numeric(0)),
        consumption = list(connections = numeric(0), weights = numeric(0)),
        information = list(connections = numeric(0), weights = numeric(0))
      )
      
      # Initialize communication protocols
      communication <<- list(
        mcp_capability = runif(1, 0.5, 1.0),
        market_signal_sensitivity = runif(1, 0.3, 1.0),
        social_communication_propensity = runif(1, 0.2, 0.8),
        costs = list(
          mcp = 0.01,
          market = 0.005,
          social = 0.002
        )
      )
      
      # Initialize learning parameters
      learning <<- list(
        performance_history = numeric(0),
        strategy_weights = switch(type,
          "rational_optimizer" = c(optimization = 1.0),
          "bounded_rational" = c(heuristic = 0.8, optimization = 0.2),
          "social_learner" = c(imitation = 0.7, own_experience = 0.3),
          "trend_follower" = c(momentum = 0.8, mean_reversion = 0.2),
          "contrarian" = c(mean_reversion = 0.8, momentum = 0.2),
          "adaptive_learner" = c(exploration = 0.3, exploitation = 0.7),
          c(default = 1.0)
        ),
        learning_rate = switch(type,
          "adaptive_learner" = 0.1,
          "social_learner" = 0.05,
          0.01
        )
      )
    },
    
    #' Make Economic Decision
    #' 
    #' @param market_data List containing market information
    #' @param network_signals List of signals from network neighbors
    #' @param aggregate_state Current aggregate economic state
    #' @return Numeric vector representing agent's decision
    make_decision = function(market_data, network_signals = list(), aggregate_state = list()) {
      "Generate economic decision based on agent type and available information"
      
      # Process information based on capacity
      processed_info <- process_information(market_data, network_signals, aggregate_state)
      
      # Type-specific decision making
      decision <- switch(type,
        "rational_optimizer" = make_optimal_decision(processed_info),
        "bounded_rational" = make_heuristic_decision(processed_info),
        "social_learner" = make_imitative_decision(processed_info, network_signals),
        "trend_follower" = make_momentum_decision(processed_info),
        "contrarian" = make_contrarian_decision(processed_info),
        "adaptive_learner" = make_adaptive_decision(processed_info),
        make_default_decision(processed_info)
      )
      
      # Apply network effects
      decision <- apply_network_effects(decision, network_signals)
      
      return(decision)
    },
    
    #' Process Available Information
    #' 
    #' @param market_data Market information
    #' @param network_signals Network neighbor signals
    #' @param aggregate_state Aggregate economic state
    #' @return Processed information within agent's capacity
    process_information = function(market_data, network_signals, aggregate_state) {
      "Process information based on agent's cognitive capacity and preferences"
      
      # Information filtering based on capacity
      capacity <- state$information_capacity
      
      # Priority-based information selection
      processed <- list()
      
      if (capacity > 0.8) {
        # High capacity: process all information
        processed$market <- market_data
        processed$network <- network_signals
        processed$aggregate <- aggregate_state
      } else if (capacity > 0.5) {
        # Medium capacity: focus on most relevant
        processed$market <- head(market_data, length(market_data) * capacity)
        processed$network <- network_signals[1:min(3, length(network_signals))]
        processed$aggregate <- aggregate_state[1:min(2, length(aggregate_state))]
      } else {
        # Low capacity: minimal processing
        processed$market <- head(market_data, 1)
        processed$network <- head(network_signals, 1)
        processed$aggregate <- head(aggregate_state, 1)
      }
      
      return(processed)
    },
    
    #' Update Agent Wealth and Learning
    #' 
    #' @param return_rate Portfolio return for this period
    update_wealth = function(return_rate) {
      "Update agent wealth and perform learning based on performance"
      
      old_wealth <- wealth
      wealth <<- wealth * (1 + return_rate)
      
      # Record performance
      performance <- return_rate
      learning$performance_history <<- c(tail(learning$performance_history, state$memory_horizon - 1), performance)
      
      # Type-specific learning
      if (type == "adaptive_learner") {
        update_adaptive_learning(performance)
      } else if (type == "social_learner") {
        update_social_learning(performance)
      }
      
      # Update network positions based on performance
      update_network_positions(performance)
    },
    
    #' Update Network Positions
    #' 
    #' @param performance Recent performance measure
    update_network_positions = function(performance) {
      "Update network positions based on performance and strategy"
      
      # Production network updates
      if (performance > 0.01) {
        # Good performance: seek more production connections
        networks$production$target_degree <<- min(10, length(networks$production$connections) + 1)
      } else if (performance < -0.01) {
        # Poor performance: reduce connections
        networks$production$target_degree <<- max(1, length(networks$production$connections) - 1)
      }
      
      # Information network updates based on type
      if (type %in% c("social_learner", "adaptive_learner")) {
        # Learning types seek more information connections
        networks$information$target_degree <<- min(15, length(networks$information$connections) + 1)
      }
    },
    
    #' Choose Communication Protocol
    #' 
    #' @param target_agent Target agent for communication
    #' @param message_type Type of message to send
    #' @return Optimal communication protocol
    choose_communication_protocol = function(target_agent, message_type) {
      "Select optimal communication protocol based on costs and effectiveness"
      
      protocols <- c("mcp", "market", "social")
      
      # Calculate expected value for each protocol
      values <- sapply(protocols, function(p) {
        effectiveness <- switch(p,
          "mcp" = communication$mcp_capability,
          "market" = communication$market_signal_sensitivity,
          "social" = communication$social_communication_propensity
        )
        cost <- communication$costs[[p]]
        return(effectiveness - cost)
      })
      
      # Select protocol with highest net value
      best_protocol <- protocols[which.max(values)]
      return(best_protocol)
    }
  )
)

# Helper functions for decision making

make_optimal_decision <- function(info) {
  # Simplified optimization (in practice would solve full optimization problem)
  if (length(info$market) > 0) {
    prices <- as.numeric(info$market[[1]])
    returns <- if(length(prices) > 1) diff(log(prices)) else 0
    decision <- sign(mean(returns, na.rm = TRUE)) * 0.5
  } else {
    decision <- 0
  }
  return(decision)
}

make_heuristic_decision <- function(info) {
  # Simple heuristic: follow recent trend
  if (length(info$market) > 0) {
    prices <- as.numeric(info$market[[1]])
    if (length(prices) >= 2) {
      recent_change <- (prices[length(prices)] - prices[length(prices)-1]) / prices[length(prices)-1]
      decision <- sign(recent_change) * 0.3
    } else {
      decision <- 0
    }
  } else {
    decision <- 0
  }
  return(decision)
}

make_imitative_decision <- function(info, network_signals) {
  # Imitate successful neighbors
  if (length(network_signals) > 0) {
    avg_signal <- mean(unlist(network_signals), na.rm = TRUE)
    decision <- avg_signal * 0.8
  } else {
    decision <- 0
  }
  return(decision)
}

make_momentum_decision <- function(info) {
  # Follow momentum
  if (length(info$market) > 0) {
    prices <- as.numeric(info$market[[1]])
    if (length(prices) >= 3) {
      momentum <- mean(diff(tail(prices, 3)), na.rm = TRUE)
      decision <- sign(momentum) * 0.4
    } else {
      decision <- 0
    }
  } else {
    decision <- 0
  }
  return(decision)
}

make_contrarian_decision <- function(info) {
  # Contrarian strategy
  if (length(info$market) > 0) {
    prices <- as.numeric(info$market[[1]])
    if (length(prices) >= 3) {
      momentum <- mean(diff(tail(prices, 3)), na.rm = TRUE)
      decision <- -sign(momentum) * 0.3  # Opposite to momentum
    } else {
      decision <- 0
    }
  } else {
    decision <- 0
  }
  return(decision)
}

make_adaptive_decision <- function(info) {
  # Adaptive learning-based decision
  # In practice would use reinforcement learning
  decision <- rnorm(1, 0, 0.2)  # Exploration
  return(decision)
}

make_default_decision <- function(info) {
  return(rnorm(1, 0, 0.1))
}

apply_network_effects <- function(decision, network_signals) {
  # Apply network influence to decision
  if (length(network_signals) > 0) {
    network_influence <- mean(unlist(network_signals), na.rm = TRUE)
    influenced_decision <- 0.7 * decision + 0.3 * network_influence
  } else {
    influenced_decision <- decision
  }
  return(influenced_decision)
}

#' Create NEURICX Agent
#' 
#' Convenience function to create a new NEURICX agent with specified parameters.
#' 
#' @param agent_id Character string for agent identifier
#' @param agent_type Agent type (rational_optimizer, bounded_rational, 
#'   social_learner, trend_follower, contrarian, adaptive_learner)
#' @param initial_wealth Initial wealth amount (default: 1,000,000)
#' 
#' @return NEURICXAgent reference class object
#' 
#' @examples
#' \dontrun{
#' agent <- create_neuricx_agent("agent_001", "rational_optimizer", 1500000)
#' }
#' 
#' @export
create_neuricx_agent <- function(agent_id, agent_type, initial_wealth = 1000000) {
  return(NEURICXAgent$new(agent_id, agent_type, initial_wealth))
}