#!/usr/bin/env Rscript


library(dplyr)
library(lubridate)
library(foreach)
library(optparse)
library(ggplot2)
library(CSHShydRology) # install manually, run the install code below after building the R environment with the FFA.yml
# https://github.com/floodnetProject16/CSHShydRology
# https://drive.google.com/file/d/1pkOSuJauiVaXAiHh_CFC1mP2GjR_VqFv/view
# install.packages("devtools")
# library(devtools)
# install_github('floodnetProject16/CSHShydRology')
# install_github('CentreForHydrology/HYDAT')
# install_github('floodnetProject16/floodnetRfa')

file_filter <- function(file) {
  if (file.exists(file)){
    return(basename(file))
  } else {
    return(NA)
  }
}
check_dir <- function(fullpath) {
    if (!dir.exists(dirname(fullpath))) {
    dir.create(dirname(fullpath), recursive = TRUE)
  }
}

find_qbfrp_Prob <-function(loess_span,nyear,peaks,qbf,thres,station_id,output_figdir) {
    # using ditribution to get QBF return period
    # make dataframe
    df <- data.frame(qpeak = peaks$discharge) %>%
      arrange(desc(qpeak)) %>%
      mutate(rank = rank(-qpeak, ties.method = "average")) %>%
      mutate(RP = (nyear + 1) / rank)
    # Check for missing values or errors
    if (any(is.na(df))) {
      print("Missing values in df:")
      print(df[is.na(df)])
      return(NA)
    }
    output_figure <- paste0(output_figdir,station_id,
      '_thres_',as.character(round(thres,2)),'.png')

    loess_span <- ifelse(is.null(loess_span), 0.5, loess_span)
    loess_fit <- loess(df[["RP"]]~qpeak, data = df, span = loess_span)
    print(paste("loess_fit$s:", loess_fit$s))
    print(paste("loess_fit$trace.hat:", loess_fit$trace.hat))
    print(paste("loess_fit$enp:", loess_fit$enp))
    print(summary(loess_fit))
    # QBF_RP <- NA
    # Check if loess fit is valid
    if (is.nan(loess_fit$s) || is.nan(loess_fit$trace.hat)) {
      print("ENTERING DEGENERATE CHECK") 
      message(paste("Degenerate loess fit at site:", station_id, "- too few observations or span too small"))
      save_empty_plot(output_figure)
      QBF_RP <- NA
      return(QBF_RP)
    }
    print("PASSED DEGENERATE CHECK") 
    withCallingHandlers(
      tryCatch({
        # print(paste0("Tried predicting at threshold ",as.character(thres))," with qbf ",as.character(qbf),", got QBF_RP ",as.character(QBF_RP))
        # print(paste("df:", df))
        QBF_RP <- predict(loess_fit, newdata = qbf)
        print(paste("QBF_RP after predict:", QBF_RP))
        # Create a data frame for the predicted values
        df$fit <- predict(loess_fit)
        # Create the plot with ggplot2
        p <- ggplot(df, aes(x = qpeak, y = RP)) +
          geom_point(color = "blue") +   # Original data points
          geom_line(aes(y = fit), color = "red") +  # Fitted LOESS curve
          labs(title =  paste0(station_id,' threshold=',as.character(round(thres,2)),' nyears=',as.character(nyear)),
          x = "Discharge (Q)",
          y = "Return Period (RP)") + 
          geom_point(aes(x = qbf, y = QBF_RP), color = "red", size = 3) +  # Mark the point
          annotate("text", x = qbf, y = QBF_RP, label = paste("RP =", round(QBF_RP,2), "\nQBF =", round(qbf, 2)),
              vjust = -1, color = "black")   # Add annotation

      # print(output_figure)
      # Save the plot to a file
      ggsave(output_figure, plot = p, width = 8, height = 6)  # Save as PNG
      }, error = function(err) {
      # Handle other potential errors here (optional)
      warning(paste("Exceedance number is too small, cannot fit loess curve at site: ", station_id, sep = ""))
      save_empty_plot(output_figure)
      }),
      warning = function(w) invokeRestart("muffleWarning")  # suppress the 50+ warnings
  )
  print(paste("QBF_RP before return:", QBF_RP))
  return (QBF_RP)
}
pred_flood_quantile_loess <-function(loess_span,nyear,peaks,qrp,thres,station_id,output_figdir) {
    # using ditribution to get flood quantile
    # make dataframe
    df <- data.frame(qpeak = peaks$discharge) %>%
      arrange(desc(qpeak)) %>%
      mutate(rank = rank(-qpeak, ties.method = "average")) %>%
      mutate(RP = (nyear + 1) / rank)
    # Check for missing values or errors
    if (any(is.na(df))) {
      print("Missing values in df:")
      print(df[is.na(df)])
      return(NA)
    }
    # thisspan <- 0.5
    loess_span <- ifelse(is.null(loess_span), 0.5, loess_span)
    # print(paste0("Function pred_flood_quantile_loess loess_span: ", loess_span))
    loess_fit <- loess(qpeak~df[["RP"]], data = df, span = loess_span)
    # print("Printing df...")
    # print(df)
    QRP <- predict(loess_fit, newdata = qrp)
    # print(paste0("Printing the return period Q from loess, ",as.character(qrp)))
    # print(QRP)
    # Create a data frame for the predicted values
    df$fit <- predict(loess_fit)
    # Create the plot with ggplot2
    p <- ggplot(df, aes(x = RP, y = qpeak)) +
      geom_point(color = "blue") +   # Original data points
      geom_line(aes(y = fit), color = "red") +  # Fitted LOESS curve
      geom_point(aes(x = qrp, y = QRP), color = "red", size = 3) +  # Mark the point
      annotate("text", x = qrp, y = QRP, label = paste("RP =", qrp, "\nQ =", round(QRP, 2)),
           vjust = -1, color = "black") +  # Add annotation
      labs(title = paste0(station_id,' threshold=',as.character(round(thres,2)),' nyears=',as.character(nyear)),
          x = "Return Period (RP)",
          y = "Discharge (Q)")
    output_figure <- paste0(output_figdir,station_id,
      '_thres_',as.character(round(thres,2)),'rp_',as.character(qrp),'.png')
    # print(output_figure)
    # Save the plot to a file
    ggsave(output_figure, plot = p, width = 8, height = 6)  # Save as PNG
    return (QRP)
}

# Function to clean and prepare the station data
clean_station_data <- function(df) {
  # Combine 'yy', 'mm', 'dd' into a Date column
  df <- df %>%
    mutate(Date = make_date(yy, mm, dd))
  
  # Filter out rows with NA in discharge
  df_filtered <- df %>%
    filter(!is.na(discharge))
  
  # Group by year and count records
  yearly_counts <- df_filtered %>%
    group_by(yy) %>%
    summarize(count = n())
  
  # Filter for years with at least 90% completeness of the year
  valid_years <- yearly_counts %>%
    filter(count >= 365.25*0.9) %>%
    pull(yy)
  
  # Filter the original dataframe for these years
  final_df <- df_filtered %>%
    filter(yy %in% valid_years)
  
  return(list(df=final_df,nyear=length(valid_years)))
}
# Function to save an empty plot
save_empty_plot <- function(filename) {
    png(filename, width = 800, height = 600)
    plot.new()
    text(0.5, 0.5, "Error: Unable to generate plot", cex = 1.5)
    dev.off()
}

get_fit <- function(station_file,discharge, final_df, thres, r0) {
  tryCatch({
    fit <- FitPot(discharge~Date, final_df, u = thres, declust = 'wrc', r = r0)
    return(fit)
    # print(fit)
  }, error = function(err) {
      message("Fit for station ",station_file," at threshold ",thres, " failed, move to next thre/station")
      return(NULL)
  })
}
# Define the function to process each station
process_station <- function(station_info,file_path,loess_span,output_csv) {

  station_file <- basename(file_path)
  dir_fig_qbf <- paste0(dirname(dirname(output_csv)),'/fig/qbf/')
  dir_fig_qrp <- paste0(dirname(dirname(output_csv)),'/fig/qrp/')
  # Check if the directory exists
  if (!dir.exists(dir_fig_qbf)) {
    # Create the directory if it doesn't exist
    dir.create(dir_fig_qbf, recursive = TRUE)
  }
  if (!dir.exists(dir_fig_qrp)) {
    # Create the directory if it doesn't exist
    dir.create(dir_fig_qrp, recursive = TRUE)
  }

  return_periods <- c(1,1.5,2,2.5,3,3.5,4,4.5,5,10) 

  stations_df = read.csv(station_info)

  i <- which(stations_df["Q_file"]==station_file)
  station_data <- stations_df[i,]
  station_id <- gsub("\\.csv$", "", station_file)
  da <- station_data$darea_snapped
  qbf_obs <- station_data$QBFobs
  percentiles <- c(0.1,0.25, 0.5, 0.75) # 
  r0 <- round(4 + log(da))
  print(paste("The minimal interval is ",r0," days.",sep=""))
  qbf_rps <- list()
  thres_rps <- list()
  flood_quantile <- list()
  # file does not exist
  if (!file.exists(file_path)) {
    
    print(paste("File does not exist or errorous, skipping:", station_file))
    qbf_rps <- list(Q_file = station_file, qbf_excess=NA,peak_excess=NA,thres = NA, qbf_rp = NA)
    for (qrp in return_periods) {
      flood_quantile[[as.character(qrp)]] <- 0
    }
    thres_rps <- rbind(thres_rps,c(qbf_rps,flood_quantile))
    write.csv(thres_rps, file = output_csv)
    print(paste(output_csv," is saved",sep=""))
  } else {
    df <- read.csv(file_path)
    final_df <- clean_station_data(df)$df     # Data cleaning steps within the function
    nyear <- clean_station_data(df)$nyear
    print(paste0("The number of years is ",as.character(nyear)))
    # calculate 
    # Loop through percentiles in parallel
    thres_list = percentiles * qbf_obs
    qbf_excess <- length(which.floodPeaks(discharge~Date,final_df, u = qbf_obs, r = r0, rlow = 0.75))
    for (thres in thres_list) {
      print(paste("Fit for station ",station_file," at threshold ",thres,sep=""))
      peaks_idx <- which.floodPeaks(discharge~Date,final_df, u = thres, r = r0, rlow = 0.75)
      peak_excess <- length(peaks_idx) # number of exceedance above threshold
      # print(paste0("The number of events exceeding the threshold is ",as.character(peak_excess)))
      flood_quantile <- list()
      if (peak_excess<=5) {
        print(paste0("The thres with error fit is: ",thres,sep=""))
        qbf_rps <- list(Q_file = station_file, qbf_excess=qbf_excess,peak_excess=peak_excess,
          thres = thres,qbf_rp = NA)
        for (qrp in return_periods) {
          flood_quantile[[as.character(qrp)]] <- NA
        }
        thres_rps <- rbind(thres_rps,c(qbf_rps,flood_quantile))
        next
      } else {
              peaks <- final_df[peaks_idx,]
              qbf_rp <- find_qbfrp_Prob(loess_span,nyear,peaks,qbf_obs,thres,station_id,dir_fig_qbf)          
              # print(paste0("The return period of QBF at threshold ",as.character(thres), " is ", as.character(qbf_rp)))
              if (is.na(qbf_rp)){
                print(paste0("Found no qbf_rp for threshold ",thres,sep=""))
                for (qrp in return_periods) {
                  flood_quantile[[as.character(qrp)]] <- NA
                }  
              } else {
                  for (qrp in return_periods) {
                    flood_quantile[[as.character(qrp)]] <- pred_flood_quantile_loess(loess_span,nyear,peaks,qrp,thres,station_id,dir_fig_qrp)
                  }  
              }
              # get the predicted flood quantile

        qbf_rps <- list(Q_file = station_file, qbf_excess=qbf_excess,peak_excess=peak_excess,
          thres = thres, qbf_rp = qbf_rp)
        thres_rps <- rbind(thres_rps,c(qbf_rps,flood_quantile))
        }
      }
    write.csv(thres_rps, file = output_csv)
    print(paste(output_csv," is saved",sep=""))
    print(file.exists(output_csv))
  }
}

# Define command-line arguments
option_list <- list(
  make_option(c("-s", "--station_info"), type="character", default=NULL, 
              help="path to station info csv file", metavar="character"),
  make_option(c("-d", "--dir_path"), type="character", default=NULL, 
              help="path to station file", metavar="character"),
  make_option(c("-l", "--loess-span"), type="numeric", default=0.5, 
            help="loess span parameter [default: %default]", metavar="numeric"),
  # make_option(c("-f", "--output_figure"), type="character", default=NULL, 
  #             help="path to output figure", metavar="character"),
  make_option(c("-f", "--output"), type="character", default=NULL, 
              help="output directory", metavar="character")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)
station_info <- opt$station_info
dir_path <- opt$dir_path
loess_span <- opt$"loess-span"
# output_figure <- opt$output_figure
output <- opt$output
# process station
print(paste0("loess_span: ", loess_span))
# process_station(station_info,file_path,loess_span,output_csv)
# get the filelist from the input file
df_files <- read.csv(station_info, header = TRUE)
valid_q_files <- df_files$Q_file[!is.na(df_files$Q_file) & df_files$Q_file != ""]
valid_stations <- sapply(file.path(dir_path, valid_q_files), file_filter)
valid_stations <- valid_stations[!is.na(valid_stations)]
station_ids <- sapply(valid_stations, function(x) gsub("\\.csv$", "", x))
print(paste0("Calculating the return period of QBF at ",as.character(length(station_ids))))
for (id in station_ids) {
  # print(id)
  file_path <- file.path(dir_path, paste(id, ".csv", sep = ""))
  print(file_path)
  # Check if the directory exists, and create it if it doesn't
  output_csv <- file.path(output, paste("csv/",id, ".csv", sep = ""))  
  check_dir(output_csv)
  process_station(station_info, file_path, loess_span, output_csv)
}
