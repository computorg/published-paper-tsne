library(plotly)
library(readr)
library(dplyr)

plot2d <- function(X, y, manifold_method) {
    fig <- plot_ly() %>%
        add_trace(x = X[, 1], y = X[, 2], color = y, colors = "Paired", type = "scatter", mode = "markers") %>%
        layout(xaxis = list(visible = FALSE), yaxis = list(visible = FALSE), legend = list(itemsizing = "constant"))
    return(fig)
}

plot3d <- function(X, y, manifold_method) {
    fig <- plot_ly() %>%
        add_trace(x = X[, 1], y = X[, 2], z = X[, 3], color = y, colors = "Paired", type="scatter3d", mode = "markers", marker = list(size = 3, opacity = 0.5)) %>%
        layout(scene = list(xaxis = list(visible = FALSE), yaxis = list(visible = FALSE), zaxis = list(visible = FALSE)), legend = list(itemsizing = "constant"))
    return(fig)
}

plot_dataset <- function(datasetname, methods, sample_size = NULL) {
    headers <- read_csv(paste0(datasetname, ".csv"), col_names = FALSE, show_col_types = FALSE, n_max = 2)
    col_names <- paste(headers[1, ], as.numeric(headers[2, ]) + 1, sep = "_")

    pandas_df <- read_csv(paste0(datasetname, ".csv"), col_names = col_names, show_col_types = FALSE, skip = 2)
    
    if (!is.null(sample_size)) {
        pandas_df <- pandas_df[sample(1:nrow(pandas_df), sample_size), ]
    }
    y <- as.factor(pandas_df$y_1)
    X <- pandas_df[, -ncol(pandas_df)]
    
    figlist <- list(1:length(methods))   
    i <- 1
    for (method in methods) {
        Xm <- X %>% select(starts_with(paste0(method,"_"))) %>% as.matrix
        ncols <- ncol(Xm)
        if (ncols == 2) {
            figlist[[i]] <- plot2d(Xm, y, method)
        } else if (ncols == 3) {
            figlist[[i]] <- plot3d(Xm, y, method)
        }
        i <- i + 1
    }
    return(figlist)
}
