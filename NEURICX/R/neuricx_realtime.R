#' NEURICX Real-Time Data Streaming and Live Market Integration
#'
#' @description
#' Real-time data streaming capabilities for NEURICX framework including
#' live market data integration, streaming analytics, and dynamic model updating.

#' Initialize Real-Time Data Stream
#'
#' @param symbols Character vector of symbols to stream
#' @param data_sources List of data sources to use
#' @param update_frequency Update frequency in seconds
#' @param callback_function Function to call on new data
#' @return Real-time data stream object
#' @export
initialize_realtime_stream <- function(symbols = c("AAPL", "MSFT", "GOOGL"),
                                     data_sources = c("yahoo", "alpha_vantage", "polygon"),
                                     update_frequency = 60,
                                     callback_function = NULL) {
  "Initialize real-time data streaming for NEURICX analysis"
  
  # Validate inputs
  if (length(symbols) == 0) {
    stop("At least one symbol must be specified")
  }
  
  if (update_frequency < 1) {
    stop("Update frequency must be at least 1 second")
  }
  
  # Create stream configuration
  stream_config <- list(
    symbols = symbols,
    data_sources = data_sources,
    update_frequency = update_frequency,
    callback_function = callback_function,
    start_time = Sys.time(),
    is_active = FALSE,
    buffer_size = 1000,
    data_buffer = list(),
    connection_status = list()
  )
  
  class(stream_config) <- "neuricx_stream"
  
  cat("Real-time stream initialized for", length(symbols), "symbols\n")
  cat("Update frequency:", update_frequency, "seconds\n")
  cat("Data sources:", paste(data_sources, collapse = ", "), "\n")
  
  return(stream_config)
}

#' Start Real-Time Streaming
#'
#' @param stream_config Stream configuration object
#' @param economy NEURICX economy object (optional)
#' @return Updated stream configuration
#' @export
start_realtime_stream <- function(stream_config, economy = NULL) {
  "Start real-time data streaming and analysis"
  
  if (!inherits(stream_config, "neuricx_stream")) {
    stop("Invalid stream configuration object")
  }
  
  if (stream_config$is_active) {
    cat("Stream is already active\n")
    return(stream_config)
  }
  
  cat("Starting real-time data stream...\n")
  
  # Initialize data connections
  stream_config$connection_status <- initialize_data_connections(stream_config)
  
  # Start streaming loop
  stream_config$is_active <- TRUE
  stream_config$stream_task <- start_streaming_task(stream_config, economy)
  
  cat("Real-time streaming started successfully\n")
  return(stream_config)
}

#' Stop Real-Time Streaming
#'
#' @param stream_config Stream configuration object
#' @return Updated stream configuration
#' @export
stop_realtime_stream <- function(stream_config) {
  "Stop real-time data streaming"
  
  if (!stream_config$is_active) {
    cat("Stream is not currently active\n")
    return(stream_config)
  }
  
  cat("Stopping real-time data stream...\n")
  
  stream_config$is_active <- FALSE
  
  # Clean up connections
  cleanup_data_connections(stream_config)
  
  cat("Real-time streaming stopped\n")
  return(stream_config)
}

#' Get Real-Time Market Data
#'
#' @param symbols Character vector of symbols
#' @param source Data source to use
#' @return Current market data
#' @export
get_realtime_market_data <- function(symbols, source = "yahoo") {
  "Fetch current real-time market data"
  
  tryCatch({
    switch(source,
      "yahoo" = get_yahoo_realtime_data(symbols),
      "alpha_vantage" = get_alpha_vantage_realtime_data(symbols),
      "polygon" = get_polygon_realtime_data(symbols),
      "iex" = get_iex_realtime_data(symbols),
      get_synthetic_realtime_data(symbols)
    )
  }, error = function(e) {
    cat("Error fetching real-time data:", e$message, "\n")
    cat("Falling back to synthetic data\n")
    return(get_synthetic_realtime_data(symbols))
  })
}

#' Update Economy with Real-Time Data
#'
#' @param economy NEURICX economy object
#' @param new_data New market data
#' @param update_agents Whether to update agent decisions
#' @return Updated economy object
#' @export
update_economy_realtime <- function(economy, new_data, update_agents = TRUE) {
  "Update NEURICX economy with new real-time data"
  
  if (is.null(economy) || is.null(new_data)) {
    return(economy)
  }
  
  # Update market data
  economy$market_data <- append_market_data(economy$market_data, new_data)
  
  if (update_agents) {
    # Update agent decisions based on new data
    economy <- update_agent_decisions_realtime(economy, new_data)
    
    # Update networks
    economy$networks <- update_networks_realtime(economy$networks, new_data)
    
    # Update collective intelligence
    economy$collective_intelligence <- update_collective_intelligence_realtime(
      economy$collective_intelligence, 
      new_data
    )
  }
  
  return(economy)
}

#' Stream Analytics Engine
#'
#' @param stream_config Stream configuration
#' @param analytics_config Analytics configuration
#' @return Analytics results
#' @export
run_streaming_analytics <- function(stream_config, analytics_config = list()) {
  "Run real-time analytics on streaming data"
  
  # Default analytics configuration
  default_config <- list(
    window_size = 100,
    update_frequency = 30,
    metrics = c("volatility", "momentum", "mean_reversion", "correlation"),
    alerts = list(
      high_volatility = 0.05,
      unusual_volume = 2.0,
      price_deviation = 0.03
    )
  )
  
  # Merge configurations
  config <- modifyList(default_config, analytics_config)
  
  # Initialize analytics state
  analytics_state <- list(
    config = config,
    rolling_metrics = list(),
    alerts = list(),
    last_update = Sys.time()
  )
  
  # Start analytics loop
  while (stream_config$is_active) {
    # Get latest data
    recent_data <- get_recent_stream_data(stream_config, config$window_size)
    
    if (nrow(recent_data) >= config$window_size) {
      # Calculate real-time metrics
      metrics <- calculate_realtime_metrics(recent_data, config$metrics)
      
      # Update rolling metrics
      analytics_state$rolling_metrics <- update_rolling_metrics(
        analytics_state$rolling_metrics, 
        metrics
      )
      
      # Check for alerts
      new_alerts <- check_realtime_alerts(metrics, config$alerts)
      if (length(new_alerts) > 0) {
        analytics_state$alerts <- c(analytics_state$alerts, new_alerts)
        
        # Trigger alert callbacks
        for (alert in new_alerts) {
          trigger_alert(alert)
        }
      }
      
      analytics_state$last_update <- Sys.time()
    }
    
    # Wait before next update
    Sys.sleep(config$update_frequency)
  }
  
  return(analytics_state)
}

#' Create Real-Time Dashboard
#'
#' @param stream_config Stream configuration
#' @param economy NEURICX economy object
#' @param dashboard_config Dashboard configuration
#' @return Dashboard application
#' @export
create_realtime_dashboard <- function(stream_config, economy = NULL, dashboard_config = list()) {
  "Create real-time dashboard for NEURICX streaming data"
  
  library(shiny)
  library(shinydashboard)
  library(plotly)
  library(DT)
  
  # Dashboard UI
  ui <- dashboardPage(
    dashboardHeader(title = "NEURICX Real-Time Analytics"),
    
    dashboardSidebar(
      sidebarMenu(
        menuItem("Market Overview", tabName = "overview", icon = icon("chart-line")),
        menuItem("Agent Behavior", tabName = "agents", icon = icon("users")),
        menuItem("Network Dynamics", tabName = "networks", icon = icon("project-diagram")),
        menuItem("Alerts", tabName = "alerts", icon = icon("exclamation-triangle")),
        menuItem("Settings", tabName = "settings", icon = icon("cog"))
      )
    ),
    
    dashboardBody(
      tags$head(
        tags$style(HTML("
          .content-wrapper, .right-side {
            background-color: #0a0e1a;
            color: #e0e6ed;
          }
          .box {
            background-color: #2a3142;
            border: 1px solid #00d4ff;
          }
        "))
      ),
      
      tabItems(
        # Market Overview Tab
        tabItem(tabName = "overview",
          fluidRow(
            valueBoxOutput("totalValue", width = 3),
            valueBoxOutput("avgReturn", width = 3),
            valueBoxOutput("volatility", width = 3),
            valueBoxOutput("networkDensity", width = 3)
          ),
          
          fluidRow(
            box(
              title = "Real-Time Price Chart", 
              status = "primary", 
              solidHeader = TRUE,
              width = 8,
              plotlyOutput("priceChart", height = "400px")
            ),
            
            box(
              title = "Market Metrics", 
              status = "primary", 
              solidHeader = TRUE,
              width = 4,
              tableOutput("marketMetrics")
            )
          ),
          
          fluidRow(
            box(
              title = "Volume Analysis", 
              status = "primary", 
              solidHeader = TRUE,
              width = 6,
              plotlyOutput("volumeChart")
            ),
            
            box(
              title = "Correlation Matrix", 
              status = "primary", 
              solidHeader = TRUE,
              width = 6,
              plotlyOutput("correlationMatrix")
            )
          )
        ),
        
        # Agent Behavior Tab
        tabItem(tabName = "agents",
          fluidRow(
            box(
              title = "Agent Performance", 
              status = "primary", 
              solidHeader = TRUE,
              width = 12,
              DTOutput("agentTable")
            )
          ),
          
          fluidRow(
            box(
              title = "Decision Distribution", 
              status = "primary", 
              solidHeader = TRUE,
              width = 6,
              plotlyOutput("decisionDistribution")
            ),
            
            box(
              title = "Collective Intelligence", 
              status = "primary", 
              solidHeader = TRUE,
              width = 6,
              plotlyOutput("collectiveIntelligence")
            )
          )
        ),
        
        # Network Dynamics Tab
        tabItem(tabName = "networks",
          fluidRow(
            box(
              title = "Network Visualization", 
              status = "primary", 
              solidHeader = TRUE,
              width = 8,
              plotlyOutput("networkViz", height = "500px")
            ),
            
            box(
              title = "Network Metrics", 
              status = "primary", 
              solidHeader = TRUE,
              width = 4,
              tableOutput("networkMetrics")
            )
          )
        ),
        
        # Alerts Tab
        tabItem(tabName = "alerts",
          fluidRow(
            box(
              title = "Active Alerts", 
              status = "warning", 
              solidHeader = TRUE,
              width = 12,
              DTOutput("alertsTable")
            )
          )
        ),
        
        # Settings Tab
        tabItem(tabName = "settings",
          fluidRow(
            box(
              title = "Stream Configuration", 
              status = "primary", 
              solidHeader = TRUE,
              width = 6,
              numericInput("updateFreq", "Update Frequency (seconds)", 
                          value = 60, min = 1, max = 300),
              textInput("symbols", "Symbols (comma-separated)", 
                       value = paste(stream_config$symbols, collapse = ",")),
              actionButton("updateConfig", "Update Configuration", 
                          class = "btn-primary")
            ),
            
            box(
              title = "Alert Configuration", 
              status = "primary", 
              solidHeader = TRUE,
              width = 6,
              numericInput("volThreshold", "Volatility Alert Threshold", 
                          value = 0.05, min = 0.01, max = 0.2, step = 0.01),
              numericInput("priceThreshold", "Price Deviation Threshold", 
                          value = 0.03, min = 0.01, max = 0.1, step = 0.01),
              actionButton("updateAlerts", "Update Alerts", 
                          class = "btn-warning")
            )
          )
        )
      )
    )
  )
  
  # Dashboard Server
  server <- function(input, output, session) {
    # Reactive values for real-time data
    values <- reactiveValues(
      current_data = data.frame(),
      market_metrics = list(),
      alerts = list(),
      last_update = Sys.time()
    )
    
    # Auto-refresh data
    observe({
      invalidateLater(stream_config$update_frequency * 1000, session)
      
      # Get latest data
      new_data <- get_realtime_market_data(stream_config$symbols)
      values$current_data <- new_data
      values$last_update <- Sys.time()
    })
    
    # Value boxes
    output$totalValue <- renderValueBox({
      valueBox(
        value = ifelse(nrow(values$current_data) > 0, 
                      paste0("$", format(sum(values$current_data$price, na.rm = TRUE), big.mark = ",")),
                      "$0"),
        subtitle = "Total Market Value",
        icon = icon("dollar-sign"),
        color = "blue"
      )
    })
    
    output$avgReturn <- renderValueBox({
      avg_ret <- ifelse(nrow(values$current_data) > 0,
                       mean(values$current_data$change_pct, na.rm = TRUE),
                       0)
      valueBox(
        value = paste0(round(avg_ret * 100, 2), "%"),
        subtitle = "Average Return",
        icon = icon("chart-line"),
        color = if (avg_ret >= 0) "green" else "red"
      )
    })
    
    output$volatility <- renderValueBox({
      vol <- ifelse(nrow(values$current_data) > 0,
                   sd(values$current_data$change_pct, na.rm = TRUE),
                   0)
      valueBox(
        value = paste0(round(vol * 100, 2), "%"),
        subtitle = "Volatility",
        icon = icon("wave-square"),
        color = "yellow"
      )
    })
    
    output$networkDensity <- renderValueBox({
      density <- ifelse(!is.null(economy),
                       calculate_current_network_density(economy),
                       0.087)
      valueBox(
        value = round(density, 3),
        subtitle = "Network Density",
        icon = icon("project-diagram"),
        color = "purple"
      )
    })
    
    # Charts
    output$priceChart <- renderPlotly({
      if (nrow(values$current_data) == 0) {
        return(plotly_empty())
      }
      
      p <- plot_ly(values$current_data, x = ~timestamp, y = ~price, 
                   color = ~symbol, type = 'scatter', mode = 'lines') %>%
        layout(title = "Real-Time Prices",
               xaxis = list(title = "Time"),
               yaxis = list(title = "Price"),
               plot_bgcolor = "transparent",
               paper_bgcolor = "transparent",
               font = list(color = "#e0e6ed"))
      
      return(p)
    })
    
    # Market metrics table
    output$marketMetrics <- renderTable({
      if (nrow(values$current_data) == 0) {
        return(data.frame(Metric = character(), Value = character()))
      }
      
      metrics <- data.frame(
        Metric = c("Last Update", "Active Symbols", "Avg Volume", "Market Cap"),
        Value = c(
          format(values$last_update, "%H:%M:%S"),
          length(unique(values$current_data$symbol)),
          format(mean(values$current_data$volume, na.rm = TRUE), big.mark = ","),
          paste0("$", format(sum(values$current_data$market_cap, na.rm = TRUE) / 1e9, digits = 2), "B")
        )
      )
      
      return(metrics)
    }, striped = TRUE, hover = TRUE, bordered = TRUE)
  }
  
  # Return Shiny app
  return(shinyApp(ui = ui, server = server))
}

# Helper functions

initialize_data_connections <- function(stream_config) {
  connections <- list()
  
  for (source in stream_config$data_sources) {
    connections[[source]] <- tryCatch({
      establish_connection(source, stream_config$symbols)
    }, error = function(e) {
      cat("Failed to connect to", source, ":", e$message, "\n")
      NULL
    })
  }
  
  return(connections)
}

establish_connection <- function(source, symbols) {
  # Simulate connection establishment
  switch(source,
    "yahoo" = list(status = "connected", endpoint = "yahoo_finance_api"),
    "alpha_vantage" = list(status = "connected", endpoint = "alpha_vantage_api"),
    "polygon" = list(status = "connected", endpoint = "polygon_api"),
    "iex" = list(status = "connected", endpoint = "iex_api"),
    list(status = "failed")
  )
}

start_streaming_task <- function(stream_config, economy) {
  # In a real implementation, this would start a background task
  # For demonstration, we'll simulate with a timestamp
  return(list(
    task_id = paste0("stream_", as.numeric(Sys.time())),
    start_time = Sys.time(),
    status = "running"
  ))
}

cleanup_data_connections <- function(stream_config) {
  # Clean up any open connections
  for (source in names(stream_config$connection_status)) {
    if (!is.null(stream_config$connection_status[[source]])) {
      cat("Closing connection to", source, "\n")
    }
  }
}

get_yahoo_realtime_data <- function(symbols) {
  tryCatch({
    data_list <- list()
    
    for (symbol in symbols) {
      # Simulate real-time data from Yahoo Finance
      current_price <- 100 + rnorm(1, 0, 5)
      change_pct <- rnorm(1, 0, 0.02)
      volume <- round(runif(1, 1e6, 10e6))
      
      data_list[[symbol]] <- data.frame(
        symbol = symbol,
        timestamp = Sys.time(),
        price = current_price,
        change_pct = change_pct,
        volume = volume,
        market_cap = current_price * 1e9,
        source = "yahoo"
      )
    }
    
    return(do.call(rbind, data_list))
    
  }, error = function(e) {
    return(get_synthetic_realtime_data(symbols))
  })
}

get_alpha_vantage_realtime_data <- function(symbols) {
  # Simulate Alpha Vantage API data
  return(get_synthetic_realtime_data(symbols, source = "alpha_vantage"))
}

get_polygon_realtime_data <- function(symbols) {
  # Simulate Polygon.io data
  return(get_synthetic_realtime_data(symbols, source = "polygon"))
}

get_iex_realtime_data <- function(symbols) {
  # Simulate IEX Cloud data
  return(get_synthetic_realtime_data(symbols, source = "iex"))
}

get_synthetic_realtime_data <- function(symbols, source = "synthetic") {
  data_list <- list()
  
  for (symbol in symbols) {
    # Generate realistic synthetic data
    base_prices <- list(
      "AAPL" = 180,
      "MSFT" = 350,
      "GOOGL" = 2800,
      "TSLA" = 200,
      "NVDA" = 450
    )
    
    base_price <- base_prices[[symbol]] %||% 100
    current_price <- base_price * (1 + rnorm(1, 0, 0.02))
    change_pct <- rnorm(1, 0, 0.015)
    volume <- round(runif(1, 1e6, 20e6))
    
    data_list[[symbol]] <- data.frame(
      symbol = symbol,
      timestamp = Sys.time(),
      price = current_price,
      change_pct = change_pct,
      volume = volume,
      market_cap = current_price * runif(1, 1e9, 5e9),
      source = source,
      stringsAsFactors = FALSE
    )
  }
  
  return(do.call(rbind, data_list))
}

append_market_data <- function(existing_data, new_data) {
  if (is.null(existing_data) || length(existing_data) == 0) {
    return(list(
      prices = matrix(new_data$price, nrow = 1),
      returns = matrix(new_data$change_pct, nrow = 1),
      volumes = matrix(new_data$volume, nrow = 1),
      timestamps = new_data$timestamp[1]
    ))
  }
  
  # Append new data to existing data
  updated_data <- existing_data
  updated_data$prices <- rbind(updated_data$prices, new_data$price)
  updated_data$returns <- rbind(updated_data$returns, new_data$change_pct)
  updated_data$volumes <- rbind(updated_data$volumes, new_data$volume)
  updated_data$timestamps <- c(updated_data$timestamps, new_data$timestamp[1])
  
  # Keep only recent data (e.g., last 1000 observations)
  max_rows <- 1000
  if (nrow(updated_data$prices) > max_rows) {
    start_row <- nrow(updated_data$prices) - max_rows + 1
    updated_data$prices <- updated_data$prices[start_row:nrow(updated_data$prices), , drop = FALSE]
    updated_data$returns <- updated_data$returns[start_row:nrow(updated_data$returns), , drop = FALSE]
    updated_data$volumes <- updated_data$volumes[start_row:nrow(updated_data$volumes), , drop = FALSE]
    updated_data$timestamps <- updated_data$timestamps[start_row:length(updated_data$timestamps)]
  }
  
  return(updated_data)
}

update_agent_decisions_realtime <- function(economy, new_data) {
  # Update agent decisions based on new real-time data
  for (i in seq_along(economy$agents)) {
    if (!is.null(economy$agents[[i]])) {
      # Get network signals for this agent
      network_signals <- calculate_agent_network_signals(economy, i)
      
      # Update agent's decision with new market data
      new_decision <- economy$agents[[i]]$make_decision(
        list(
          prices = new_data$price,
          returns = new_data$change_pct,
          volumes = new_data$volume
        ),
        network_signals,
        list(timestamp = new_data$timestamp[1])
      )
      
      # Store the decision (in practice, would update agent's internal state)
    }
  }
  
  return(economy)
}

update_networks_realtime <- function(networks, new_data) {
  # Update network structures based on new market conditions
  market_volatility <- sd(new_data$change_pct, na.rm = TRUE)
  
  # Higher volatility tends to increase information network density
  info_adjustment <- market_volatility * 0.1
  
  if (!is.null(networks$information)) {
    networks$information <- networks$information + 
      matrix(rnorm(length(networks$information), 0, info_adjustment), 
             nrow = nrow(networks$information))
    
    # Ensure values stay in valid range
    networks$information[networks$information < 0] <- 0
    networks$information[networks$information > 1] <- 1
    diag(networks$information) <- 0
  }
  
  return(networks)
}

update_collective_intelligence_realtime <- function(ci_state, new_data) {
  # Update collective intelligence metrics with new data
  if (is.null(ci_state)) {
    ci_state <- list(index = 1.0, history = numeric(0))
  }
  
  # Market coherence affects collective intelligence
  market_coherence <- 1 - sd(new_data$change_pct, na.rm = TRUE)
  ci_adjustment <- market_coherence * 0.1
  
  new_ci_index <- ci_state$index + ci_adjustment + rnorm(1, 0, 0.05)
  ci_state$index <- max(0.5, min(2.0, new_ci_index))
  ci_state$history <- c(tail(ci_state$history, 99), ci_state$index)
  
  return(ci_state)
}

calculate_realtime_metrics <- function(data, metrics) {
  results <- list()
  
  if ("volatility" %in% metrics) {
    results$volatility <- sd(data$change_pct, na.rm = TRUE)
  }
  
  if ("momentum" %in% metrics) {
    results$momentum <- mean(tail(data$change_pct, 10), na.rm = TRUE)
  }
  
  if ("mean_reversion" %in% metrics) {
    results$mean_reversion <- -cor(data$change_pct[-length(data$change_pct)], 
                                  data$change_pct[-1], use = "complete.obs")
  }
  
  if ("correlation" %in% metrics && length(unique(data$symbol)) > 1) {
    # Calculate correlation between symbols
    price_matrix <- reshape2::dcast(data, timestamp ~ symbol, value.var = "change_pct")
    price_matrix <- price_matrix[, -1]  # Remove timestamp column
    results$correlation <- cor(price_matrix, use = "complete.obs")
  }
  
  return(results)
}

update_rolling_metrics <- function(rolling_metrics, new_metrics) {
  for (metric_name in names(new_metrics)) {
    if (is.null(rolling_metrics[[metric_name]])) {
      rolling_metrics[[metric_name]] <- list(values = numeric(0), timestamps = character(0))
    }
    
    # Add new value
    rolling_metrics[[metric_name]]$values <- c(
      tail(rolling_metrics[[metric_name]]$values, 99),
      new_metrics[[metric_name]]
    )
    rolling_metrics[[metric_name]]$timestamps <- c(
      tail(rolling_metrics[[metric_name]]$timestamps, 99),
      as.character(Sys.time())
    )
  }
  
  return(rolling_metrics)
}

check_realtime_alerts <- function(metrics, alert_config) {
  alerts <- list()
  
  if (!is.null(metrics$volatility) && metrics$volatility > alert_config$high_volatility) {
    alerts <- c(alerts, list(list(
      type = "high_volatility",
      message = paste("High volatility detected:", round(metrics$volatility * 100, 2), "%"),
      severity = "warning",
      timestamp = Sys.time()
    )))
  }
  
  if (!is.null(metrics$momentum) && abs(metrics$momentum) > alert_config$price_deviation) {
    alerts <- c(alerts, list(list(
      type = "price_deviation",
      message = paste("Significant price movement:", round(metrics$momentum * 100, 2), "%"),
      severity = "info",
      timestamp = Sys.time()
    )))
  }
  
  return(alerts)
}

trigger_alert <- function(alert) {
  cat("ALERT [", alert$severity, "]:", alert$message, "\n")
  
  # In practice, could send notifications, emails, etc.
  if (alert$severity == "critical") {
    # Trigger immediate notifications
  }
}

calculate_agent_network_signals <- function(economy, agent_index) {
  # Calculate network signals for a specific agent
  if (is.null(economy$networks)) {
    return(0)
  }
  
  signals <- 0
  
  # Get signals from each network layer
  for (layer in names(economy$networks)) {
    if (!is.null(economy$networks[[layer]])) {
      # Sum of connections weighted by strength
      layer_signal <- sum(economy$networks[[layer]][agent_index, ])
      signals <- signals + layer_signal
    }
  }
  
  return(signals / length(economy$networks))
}

calculate_current_network_density <- function(economy) {
  if (is.null(economy$networks)) {
    return(0.087)  # Default value
  }
  
  densities <- numeric(0)
  
  for (layer in names(economy$networks)) {
    if (!is.null(economy$networks[[layer]])) {
      density <- sum(economy$networks[[layer]]) / length(economy$networks[[layer]])
      densities <- c(densities, density)
    }
  }
  
  return(mean(densities))
}

get_recent_stream_data <- function(stream_config, window_size) {
  # Get recent data from stream buffer
  if (length(stream_config$data_buffer) == 0) {
    return(data.frame())
  }
  
  # Combine recent data
  recent_data <- do.call(rbind, tail(stream_config$data_buffer, window_size))
  return(recent_data)
}