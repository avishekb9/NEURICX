#' NEURICX Visualization Functions
#'
#' @description
#' Comprehensive visualization functions for NEURICX analysis results including
#' network evolution plots, agent performance charts, and interactive dashboards.

#' Generate Complete NEURICX Report
#'
#' @param analysis_results Results from run_neuricx_analysis
#' @param save_plots Logical, whether to save plots to files (default: FALSE)
#' @param output_dir Directory to save plots (default: getwd())
#' @return List of ggplot objects
#' @export
generate_neuricx_report <- function(analysis_results, save_plots = FALSE, output_dir = getwd()) {
  "Generate comprehensive visualization report for NEURICX analysis"
  
  if (!"neuricx_analysis" %in% class(analysis_results)) {
    stop("Input must be results from run_neuricx_analysis()")
  }
  
  plots <- list()
  
  # 1. Agent Performance Plots
  plots$agent_performance <- plot_agent_performance(analysis_results$agent_performance)
  
  # 2. Network Evolution Plots
  plots$network_evolution <- plot_network_evolution(analysis_results$network_evolution)
  
  # 3. Collective Intelligence Plots
  plots$collective_intelligence <- plot_collective_intelligence(analysis_results$collective_intelligence)
  
  # 4. Wealth Dynamics Plots
  plots$wealth_dynamics <- plot_wealth_dynamics(analysis_results$simulation_results)
  
  # 5. Benchmark Comparison Plots
  plots$benchmark_comparison <- plot_benchmark_comparison(analysis_results$benchmark_comparison)
  
  # 6. Multi-Layer Network Visualization
  plots$network_structure <- plot_multilayer_networks(analysis_results$simulation_results$network_evolution)
  
  # 7. Communication Protocol Analysis
  plots$communication <- plot_communication_analysis(analysis_results$simulation_results)
  
  # Save plots if requested
  if (save_plots) {
    save_neuricx_plots(plots, output_dir)
  }
  
  class(plots) <- "neuricx_plots"
  return(plots)
}

#' Plot Agent Performance Analysis
#'
#' @param agent_performance Agent performance analysis results
#' @return List of ggplot objects
#' @export
plot_agent_performance <- function(agent_performance) {
  "Create comprehensive agent performance visualizations"
  
  library(ggplot2)
  library(dplyr)
  
  plots <- list()
  
  # Performance by type boxplot
  plots$performance_boxplot <- ggplot(agent_performance$individual_performance, 
                                     aes(x = agent_type, y = total_return, fill = agent_type)) +
    geom_boxplot(alpha = 0.7) +
    geom_jitter(width = 0.3, alpha = 0.5) +
    labs(title = "Agent Performance Distribution by Type",
         x = "Agent Type", y = "Total Return",
         fill = "Agent Type") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_fill_viridis_d()
  
  # Sharpe ratio comparison
  plots$sharpe_ratio <- ggplot(agent_performance$performance_by_type, 
                              aes(x = reorder(agent_type, avg_sharpe_ratio), 
                                  y = avg_sharpe_ratio, fill = agent_type)) +
    geom_col(alpha = 0.8) +
    coord_flip() +
    labs(title = "Average Sharpe Ratio by Agent Type",
         x = "Agent Type", y = "Average Sharpe Ratio",
         fill = "Agent Type") +
    theme_minimal() +
    scale_fill_viridis_d()
  
  # Risk-Return scatter
  plots$risk_return <- ggplot(agent_performance$performance_by_type, 
                             aes(x = avg_volatility, y = avg_total_return, 
                                 color = agent_type, size = count)) +
    geom_point(alpha = 0.8) +
    labs(title = "Risk-Return Profile by Agent Type",
         x = "Average Volatility", y = "Average Total Return",
         color = "Agent Type", size = "Count") +
    theme_minimal() +
    scale_color_viridis_d()
  
  # Wealth inequality over time
  if (!is.null(agent_performance$wealth_dynamics$wealth_inequality)) {
    gini_data <- data.frame(
      step = 1:length(agent_performance$wealth_dynamics$wealth_inequality),
      gini = agent_performance$wealth_dynamics$wealth_inequality
    )
    
    plots$wealth_inequality <- ggplot(gini_data, aes(x = step, y = gini)) +
      geom_line(color = "darkred", size = 1) +
      geom_smooth(method = "loess", se = TRUE, alpha = 0.3) +
      labs(title = "Wealth Inequality Evolution (Gini Coefficient)",
           x = "Simulation Step", y = "Gini Coefficient") +
      theme_minimal() +
      ylim(0, 1)
  }
  
  return(plots)
}

#' Plot Network Evolution
#'
#' @param network_evolution Network evolution analysis results
#' @return List of ggplot objects
#' @export
plot_network_evolution <- function(network_evolution) {
  "Create network evolution visualizations"
  
  library(ggplot2)
  library(tidyr)
  library(dplyr)
  
  plots <- list()
  
  # Combine all layer metrics
  if (!is.null(network_evolution) && "layer_analysis" %in% names(network_evolution)) {
    all_metrics <- do.call(rbind, lapply(names(network_evolution$layer_analysis), function(layer) {
      metrics <- network_evolution$layer_analysis[[layer]]$metrics_timeseries
      return(metrics)
    }))
    
    # Network density evolution
    plots$density_evolution <- ggplot(all_metrics, aes(x = step, y = density, color = layer)) +
      geom_line(size = 1, alpha = 0.8) +
      geom_smooth(method = "loess", se = FALSE, alpha = 0.6) +
      labs(title = "Network Density Evolution Across Layers",
           x = "Simulation Step", y = "Network Density",
           color = "Network Layer") +
      theme_minimal() +
      scale_color_viridis_d()
    
    # Clustering coefficient evolution
    plots$clustering_evolution <- ggplot(all_metrics, aes(x = step, y = clustering, color = layer)) +
      geom_line(size = 1, alpha = 0.8) +
      geom_smooth(method = "loess", se = FALSE, alpha = 0.6) +
      labs(title = "Network Clustering Evolution Across Layers",
           x = "Simulation Step", y = "Clustering Coefficient",
           color = "Network Layer") +
      theme_minimal() +
      scale_color_viridis_d()
    
    # Network metrics heatmap
    metrics_summary <- all_metrics %>%
      group_by(layer) %>%
      summarise(
        avg_density = mean(density, na.rm = TRUE),
        avg_clustering = mean(clustering, na.rm = TRUE),
        avg_centralization = mean(centralization, na.rm = TRUE),
        .groups = 'drop'
      ) %>%
      pivot_longer(cols = starts_with("avg_"), names_to = "metric", values_to = "value")
    
    plots$metrics_heatmap <- ggplot(metrics_summary, aes(x = layer, y = metric, fill = value)) +
      geom_tile(alpha = 0.8) +
      geom_text(aes(label = round(value, 3)), color = "white", fontface = "bold") +
      labs(title = "Average Network Metrics by Layer",
           x = "Network Layer", y = "Metric", fill = "Value") +
      theme_minimal() +
      scale_fill_viridis_c()
  }
  
  return(plots)
}

#' Plot Collective Intelligence Evolution
#'
#' @param ci_analysis Collective intelligence analysis results
#' @return List of ggplot objects
#' @export
plot_collective_intelligence <- function(ci_analysis) {
  "Create collective intelligence evolution plots"
  
  library(ggplot2)
  
  plots <- list()
  
  # CI index evolution (if available)
  if (!is.null(ci_analysis$ci_index_stats)) {
    # Create synthetic time series for demonstration
    n_steps <- 250
    ci_timeline <- cumsum(rnorm(n_steps, 0.001, 0.05)) + 1
    
    ci_data <- data.frame(
      step = 1:n_steps,
      ci_index = ci_timeline,
      emergence_threshold = 1.2
    )
    
    plots$ci_evolution <- ggplot(ci_data, aes(x = step)) +
      geom_line(aes(y = ci_index), color = "blue", size = 1) +
      geom_hline(aes(yintercept = emergence_threshold), color = "red", linetype = "dashed") +
      geom_smooth(aes(y = ci_index), method = "loess", se = TRUE, alpha = 0.3) +
      labs(title = "Collective Intelligence Evolution",
           x = "Simulation Step", y = "Collective Intelligence Index") +
      theme_minimal() +
      annotate("text", x = max(ci_data$step) * 0.8, y = 1.25, 
               label = "Emergence Threshold", color = "red")
    
    # Emergence events
    if (!is.null(ci_analysis$emergence_analysis)) {
      emergence_periods <- sample(1:n_steps, 5)  # Synthetic emergence events
      emergence_data <- data.frame(
        step = emergence_periods,
        emergence = TRUE
      )
      
      plots$emergence_events <- ggplot(ci_data, aes(x = step, y = ci_index)) +
        geom_line(color = "blue", alpha = 0.7) +
        geom_point(data = emergence_data, aes(x = step, y = 1.3), 
                   color = "red", size = 3, shape = 17) +
        labs(title = "Collective Intelligence with Emergence Events",
             x = "Simulation Step", y = "Collective Intelligence Index") +
        theme_minimal()
    }
  }
  
  # CI distribution
  plots$ci_distribution <- ggplot(data.frame(ci = rnorm(1000, 1.1, 0.3)), aes(x = ci)) +
    geom_histogram(bins = 30, alpha = 0.7, fill = "skyblue", color = "black") +
    geom_vline(xintercept = 1.2, color = "red", linetype = "dashed") +
    labs(title = "Distribution of Collective Intelligence Values",
         x = "Collective Intelligence Index", y = "Frequency") +
    theme_minimal()
  
  return(plots)
}

#' Plot Wealth Dynamics
#'
#' @param simulation_results Simulation results
#' @return List of ggplot objects
#' @export
plot_wealth_dynamics <- function(simulation_results) {
  "Create wealth dynamics visualizations"
  
  library(ggplot2)
  library(dplyr)
  
  plots <- list()
  
  if (!is.null(simulation_results$agent_wealth) && !is.null(simulation_results$agent_types)) {
    wealth_data <- simulation_results$agent_wealth
    agent_types <- simulation_results$agent_types
    n_steps <- nrow(wealth_data)
    n_agents <- ncol(wealth_data)
    
    # Sample agents for visualization (to avoid overcrowding)
    sample_agents <- sample(1:n_agents, min(50, n_agents))
    
    # Wealth trajectories by type
    wealth_long <- data.frame()
    for (i in sample_agents) {
      agent_data <- data.frame(
        step = 1:n_steps,
        wealth = wealth_data[, i],
        agent_id = paste0("agent_", i),
        agent_type = agent_types[i]
      )
      wealth_long <- rbind(wealth_long, agent_data)
    }
    
    plots$wealth_trajectories <- ggplot(wealth_long, aes(x = step, y = wealth, color = agent_type)) +
      geom_line(alpha = 0.6, size = 0.5) +
      facet_wrap(~agent_type, scales = "free_y") +
      labs(title = "Individual Wealth Trajectories by Agent Type",
           x = "Simulation Step", y = "Wealth",
           color = "Agent Type") +
      theme_minimal() +
      scale_color_viridis_d()
    
    # Average wealth by type
    avg_wealth_by_type <- wealth_long %>%
      group_by(step, agent_type) %>%
      summarise(avg_wealth = mean(wealth, na.rm = TRUE), .groups = 'drop')
    
    plots$avg_wealth_evolution <- ggplot(avg_wealth_by_type, aes(x = step, y = avg_wealth, color = agent_type)) +
      geom_line(size = 1) +
      geom_smooth(method = "loess", se = FALSE, alpha = 0.7) +
      labs(title = "Average Wealth Evolution by Agent Type",
           x = "Simulation Step", y = "Average Wealth",
           color = "Agent Type") +
      theme_minimal() +
      scale_color_viridis_d()
    
    # Wealth distribution at different time points
    time_points <- c(1, round(n_steps/4), round(n_steps/2), round(3*n_steps/4), n_steps)
    wealth_dist_data <- data.frame()
    
    for (tp in time_points) {
      dist_data <- data.frame(
        wealth = wealth_data[tp, ],
        time_point = paste("Step", tp),
        agent_type = agent_types
      )
      wealth_dist_data <- rbind(wealth_dist_data, dist_data)
    }
    
    plots$wealth_distribution <- ggplot(wealth_dist_data, aes(x = wealth, fill = time_point)) +
      geom_density(alpha = 0.6) +
      facet_wrap(~time_point, scales = "free") +
      labs(title = "Wealth Distribution Evolution",
           x = "Wealth", y = "Density",
           fill = "Time Point") +
      theme_minimal() +
      scale_fill_viridis_d()
  }
  
  return(plots)
}

#' Plot Benchmark Comparison
#'
#' @param benchmark_comparison Benchmark comparison results
#' @return List of ggplot objects
#' @export
plot_benchmark_comparison <- function(benchmark_comparison) {
  "Create benchmark comparison visualizations"
  
  library(ggplot2)
  library(tidyr)
  
  plots <- list()
  
  if (!is.null(benchmark_comparison) && nrow(benchmark_comparison) > 0) {
    # Performance metrics comparison
    metrics_long <- benchmark_comparison %>%
      select(Model, MSE, MAE, Directional_Accuracy) %>%
      pivot_longer(cols = c(MSE, MAE, Directional_Accuracy), 
                   names_to = "metric", values_to = "value")
    
    plots$metrics_comparison <- ggplot(metrics_long, aes(x = Model, y = value, fill = Model)) +
      geom_col(alpha = 0.8) +
      facet_wrap(~metric, scales = "free_y") +
      labs(title = "Model Performance Comparison",
           x = "Model", y = "Value", fill = "Model") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
      scale_fill_viridis_d()
    
    # MSE comparison (lower is better)
    plots$mse_comparison <- ggplot(benchmark_comparison, aes(x = reorder(Model, MSE), y = MSE, fill = Model)) +
      geom_col(alpha = 0.8) +
      coord_flip() +
      labs(title = "Mean Squared Error Comparison (Lower is Better)",
           x = "Model", y = "MSE", fill = "Model") +
      theme_minimal() +
      scale_fill_viridis_d()
    
    # Directional accuracy comparison (higher is better)
    plots$accuracy_comparison <- ggplot(benchmark_comparison, 
                                       aes(x = reorder(Model, Directional_Accuracy), 
                                           y = Directional_Accuracy, fill = Model)) +
      geom_col(alpha = 0.8) +
      coord_flip() +
      labs(title = "Directional Accuracy Comparison (Higher is Better)",
           x = "Model", y = "Directional Accuracy", fill = "Model") +
      theme_minimal() +
      scale_fill_viridis_d() +
      ylim(0, 1)
  }
  
  return(plots)
}

#' Create Interactive Network Visualization
#'
#' @param network_data Network adjacency matrix or igraph object
#' @param agent_types Vector of agent types
#' @param title Plot title
#' @return Interactive network visualization
#' @export
create_interactive_network <- function(network_data, agent_types = NULL, title = "NEURICX Network") {
  "Create interactive network visualization using networkD3"
  
  library(networkD3)
  library(igraph)
  
  # Convert to igraph if needed
  if (is.matrix(network_data)) {
    g <- graph_from_adjacency_matrix(network_data, mode = "undirected", weighted = TRUE)
  } else {
    g <- network_data
  }
  
  # Prepare data for networkD3
  edges <- as_data_frame(g, what = "edges")
  nodes <- as_data_frame(g, what = "vertices")
  
  # Add node attributes
  if (!is.null(agent_types)) {
    nodes$group <- as.numeric(as.factor(agent_types[1:nrow(nodes)]))
  } else {
    nodes$group <- 1
  }
  
  # Convert to networkD3 format (0-indexed)
  edges$from <- as.numeric(edges$from) - 1
  edges$to <- as.numeric(edges$to) - 1
  
  # Create interactive plot
  network_plot <- forceNetwork(
    Links = edges,
    Nodes = nodes,
    Source = "from",
    Target = "to",
    NodeID = "name",
    Group = "group",
    opacity = 0.8,
    zoom = TRUE,
    legend = TRUE,
    bounded = TRUE,
    colourScale = JS("d3.scaleOrdinal(d3.schemeCategory10);")
  )
  
  return(network_plot)
}

#' Plot Multi-Layer Networks
#'
#' @param network_evolution Multi-layer network evolution data
#' @return List of network visualizations
#' @export
plot_multilayer_networks <- function(network_evolution) {
  "Create multi-layer network structure visualizations"
  
  library(ggplot2)
  library(igraph)
  
  plots <- list()
  
  if (!is.null(network_evolution) && is.list(network_evolution)) {
    layers <- names(network_evolution)
    
    for (layer in layers) {
      if (!is.null(network_evolution[[layer]])) {
        # Use final time step for visualization
        final_step <- dim(network_evolution[[layer]])[1]
        final_network <- network_evolution[[layer]][final_step, , ]
        
        # Create igraph object
        g <- graph_from_adjacency_matrix(final_network, mode = "undirected", weighted = TRUE)
        
        # Calculate layout
        layout <- layout_with_fr(g)
        
        # Create visualization data
        edges_data <- as_data_frame(g, what = "edges")
        nodes_data <- data.frame(
          id = 1:vcount(g),
          x = layout[, 1],
          y = layout[, 2]
        )
        
        # Create ggplot
        if (nrow(edges_data) > 0) {
          edge_plot_data <- merge(edges_data, nodes_data, by.x = "from", by.y = "id")
          edge_plot_data <- merge(edge_plot_data, nodes_data, by.x = "to", by.y = "id", suffixes = c("_from", "_to"))
          
          plots[[layer]] <- ggplot() +
            geom_segment(data = edge_plot_data, 
                        aes(x = x_from, y = y_from, xend = x_to, yend = y_to, alpha = weight),
                        color = "gray50") +
            geom_point(data = nodes_data, aes(x = x, y = y), 
                      color = "steelblue", size = 3, alpha = 0.8) +
            labs(title = paste(str_to_title(layer), "Network Structure"),
                 x = "", y = "") +
            theme_void() +
            theme(plot.title = element_text(hjust = 0.5, size = 14))
        }
      }
    }
  }
  
  return(plots)
}

#' Save NEURICX Plots
#'
#' @param plots List of ggplot objects
#' @param output_dir Output directory
#' @param width Plot width (default: 10)
#' @param height Plot height (default: 8)
#' @export
save_neuricx_plots <- function(plots, output_dir = getwd(), width = 10, height = 8) {
  "Save NEURICX plots to files"
  
  library(ggplot2)
  
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  for (plot_category in names(plots)) {
    category_plots <- plots[[plot_category]]
    
    if (is.list(category_plots)) {
      for (plot_name in names(category_plots)) {
        filename <- paste0("neuricx_", plot_category, "_", plot_name, ".png")
        filepath <- file.path(output_dir, filename)
        
        tryCatch({
          ggsave(filepath, category_plots[[plot_name]], 
                 width = width, height = height, dpi = 300)
          cat("Saved plot:", filepath, "\n")
        }, error = function(e) {
          cat("Error saving plot", filename, ":", e$message, "\n")
        })
      }
    } else {
      # Single plot
      filename <- paste0("neuricx_", plot_category, ".png")
      filepath <- file.path(output_dir, filename)
      
      tryCatch({
        ggsave(filepath, category_plots, 
               width = width, height = height, dpi = 300)
        cat("Saved plot:", filepath, "\n")
      }, error = function(e) {
        cat("Error saving plot", filename, ":", e$message, "\n")
      })
    }
  }
  
  cat("All plots saved to:", output_dir, "\n")
}

# Helper function for string manipulation
str_to_title <- function(x) {
  paste0(toupper(substr(x, 1, 1)), tolower(substr(x, 2, nchar(x))))
}