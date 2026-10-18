## ============================================================
## NHPP Simulation: RSF vs Logistic Regression
## Under Varying Temporal Structure (Daily Cycle Amplitude)
## ============================================================
##
## DGP follows Maurizio's specification:
##   lambda_i(t) = lambda_0(t) * exp(x_i' * beta)
##
## where lambda_0(t) = lambda_base * [1 + a * sin(2*pi*(t - phi)/24)]
##
## Amplitude 'a' is varied across four levels (0, 0.3, 0.6, 0.8)
## while the feature signal (beta) remains constant.
##
## Parameters calibrated to match ULB downsampled dataset:
##   n = 50,000, target fraud prevalence = 492/50492 ~ 0.97%
##   48-hour observation window, phi = 2 (peak at 8am)
##   p = 30 covariates (8 informative, 22 null)
## ============================================================

library(caret)
library(dplyr)
library(smotefamily)
library(randomForestSRC)
library(pROC)
library(PRROC)
library(ggplot2)
library(tidyr)
library(pander)

## ----------------------------------------------------------
## 1. Parameters
## ----------------------------------------------------------

set.seed(44940076)

n           <- 50000L          # match downsampled ULB size
p           <- 30L             # match ULB covariate count (Maurizio spec)
days        <- 2               # 48-hour window matching ULB
target_rate <- 492 / 50492     # match downsampled fraud prevalence (~0.97%)
phi         <- 2               # phase shift: peak occurs at phi + 6 = 8am
n_splits    <- 10L             # repeated splits per amplitude profile

# Beta: 8 informative covariates, 22 null (matching Maurizio's specification)
# Positive and negative effects of varying strengths
beta <- c(
  1.2,  0.9,  0.7,  0.5,  0.3,   # strong-to-weak positive effects
  -0.5, -0.7, -1.0,                # negative effects
  rep(0, p - 8)                   # 22 null covariates
)

# Amplitude profiles: controls strength of daily fraud cycle
# a = 0: flat baseline, LR is the correctly specified model (theoretical anchor)
# a = 0.8: strong daily cycle matching Maurizio's original specification
amplitude_profiles <- list(
  none     = 0.0,
  mild     = 0.3,
  moderate = 0.6,
  strong   = 0.8
)

## ----------------------------------------------------------
## 2. NHPP Data Generation Function
## ----------------------------------------------------------
## Generates transactions where:
##   P(fraud | X, t) = lambda_base * lambda_0(t) * exp(X' * beta)
##
## lambda_0(t): daily cycle baseline (sine function of hour_of_day)
## exp(X'*beta): feature-based risk multiplier
## lambda_base: calibrated so mean fraud rate matches target_rate
##
## Fraud labels generated via rbinom — stochastic, not pre-fixed.
## This is more statistically honest than sampling exactly n_fraud rows.
##
## Note on phi: sin(2*pi*(t - phi)/24) peaks at t = phi + 6.
## With phi = 2, the peak falls at hour 8 (8am), which is
## documented and realistic for transaction fraud patterns.

generate_nhpp_data <- function(n, p, target_rate, days, phi, a, beta,
                               seed = 44940076) {
  set.seed(seed)
  
  # Transaction timestamps over observation window
  time_hours  <- runif(n, 0, 24 * days)
  hour_of_day <- time_hours %% 24
  
  # Daily cycle baseline — ensures positivity via pmax
  daily_cycle <- 1 + a * sin(2 * pi * (hour_of_day - phi) / 24)
  daily_cycle <- pmax(daily_cycle, 0.05)
  
  # Covariates: p columns, standard normal
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  colnames(X) <- paste0("x", 1:p)
  
  # Linear predictor and risk multiplier from feature space
  linpred         <- as.numeric(X %*% beta)
  risk_multiplier <- exp(linpred)
  
  # Calibrate base rate to match target prevalence
  lambda_base <- target_rate / mean(daily_cycle * risk_multiplier)
  
  # Individual fraud probabilities — capped at 0.5 for numerical safety
  prob_fraud <- lambda_base * daily_cycle * risk_multiplier
  prob_fraud <- pmin(prob_fraud, 0.5)
  
  # Generate fraud labels via Binomial sampling (stochastic count)
  fraud <- rbinom(n, 1, prob_fraud)
  
  # Construct dataset
  df <- data.frame(X)
  df$hour_of_day <- hour_of_day  # temporal feature: available to both LR and RSF
  df$time        <- 1:n          # proxy row index: survival time for RSF
  df$status      <- as.integer(fraud)
  
  df
}

## ----------------------------------------------------------
## 3. Model Evaluation Loop
## ----------------------------------------------------------

message("=== Starting NHPP Simulation ===")
message("Amplitude profiles: ",
        paste(names(amplitude_profiles), unlist(amplitude_profiles),
              sep = " = ", collapse = ", "))
message("Splits per profile: ", n_splits)
message("")

sim_results_store <- list()

for (profile_name in names(amplitude_profiles)) {
  
  a_val <- amplitude_profiles[[profile_name]]
  message("--- Profile: ", profile_name, " (a = ", a_val, ") ---")
  
  sim_results_store[[profile_name]] <- list(
    auprc_lr        = numeric(n_splits),
    auprc_rsf       = numeric(n_splits),
    auprc_rsf_smote = numeric(n_splits),
    auc_lr          = numeric(n_splits),
    auc_rsf         = numeric(n_splits),
    auc_rsf_smote   = numeric(n_splits)
  )
  
  # Generate one dataset per amplitude profile
  sim_data <- generate_nhpp_data(
    n = n, p = p, target_rate = target_rate,
    days = days, phi = phi, a = a_val,
    beta = beta, seed = 44940076
  )
  
  message("  Fraud cases generated: ", sum(sim_data$status),
          " (target: ~", round(n * target_rate), ")")
  
  for (i in seq_len(n_splits)) {
    message("  Split ", i, " of ", n_splits, " — ", Sys.time())
    
    # Stratified 70-30 split
    sim_train_idx <- createDataPartition(sim_data$status, p = 0.7, list = FALSE)
    sim_train <- sim_data[sim_train_idx, ]
    sim_test  <- sim_data[-sim_train_idx, ]
    
    sim_train$status <- as.numeric(as.character(sim_train$status))
    sim_test$status  <- as.numeric(as.character(sim_test$status))
    sim_true_labels  <- sim_test$status
    
    # Safety check: confirm fraud cases present in both subsets
    stopifnot(
      sum(sim_train$status) > 0,
      sum(sim_test$status)  > 0
    )
    
    ## ==========================================
    ## LOGISTIC REGRESSION
    ## Features: x1...x30 + hour_of_day (linear)
    ## LR gets direct access to the temporal signal
    ## but can only model it as a linear covariate
    ## ==========================================
    
    lr_cols       <- c(paste0("x", 1:p), "hour_of_day")
    sim_lr_formula <- as.formula(
      paste("status ~", paste(lr_cols, collapse = " + "))
    )
    
    sim_lr_model <- glm(sim_lr_formula, data = sim_train, family = binomial)
    sim_lr_prob  <- predict(sim_lr_model, newdata = sim_test, type = "response")
    
    sim_lr_auc  <- as.numeric(auc(roc(
      response  = sim_true_labels,
      predictor = sim_lr_prob,
      quiet     = TRUE
    )))
    
    sim_lr_keep <- is.finite(sim_lr_prob) & is.finite(sim_true_labels)
    sim_lr_pr   <- pr.curve(
      scores.class0 = sim_lr_prob[sim_lr_keep][sim_true_labels[sim_lr_keep] == 1],
      scores.class1 = sim_lr_prob[sim_lr_keep][sim_true_labels[sim_lr_keep] == 0],
      curve = FALSE
    )
    
    sim_results_store[[profile_name]]$auprc_lr[i] <- sim_lr_pr$auc.integral
    sim_results_store[[profile_name]]$auc_lr[i]   <- sim_lr_auc
    
    ## ==========================================
    ## RSF (No SMOTE)
    ## Features: x1...x30 + hour_of_day
    ## Survival time: proxy row index (time)
    ## RSF can model nonlinear patterns in
    ## hour_of_day via tree splits, unlike LR
    ## ==========================================
    
    rsf_cols      <- c(paste0("x", 1:p), "hour_of_day", "time", "status")
    sim_rsf_train <- sim_train[, rsf_cols]
    sim_rsf_test  <- sim_test[,  rsf_cols]
    
    sim_rsf_model <- rfsrc(
      Surv(time, status) ~ .,
      data      = sim_rsf_train,
      ntree     = 100,
      nodesize  = 15,
      splitrule = "logrank",
      nthread   = 4,
      fast      = TRUE
    )
    
    sim_rsf_pred <- predict(
      sim_rsf_model,
      newdata = sim_rsf_test[, sim_rsf_model$xvar.names, drop = FALSE]
    )
    sim_rsf_risk <- as.numeric(
      1 - sim_rsf_pred$survival[, ncol(sim_rsf_pred$survival)]
    )
    
    sim_rsf_auc  <- as.numeric(auc(roc(
      response  = sim_true_labels,
      predictor = sim_rsf_risk,
      quiet     = TRUE
    )))
    
    sim_rsf_keep <- is.finite(sim_rsf_risk) & is.finite(sim_true_labels)
    sim_rsf_pr   <- pr.curve(
      scores.class0 = sim_rsf_risk[sim_rsf_keep][sim_true_labels[sim_rsf_keep] == 1],
      scores.class1 = sim_rsf_risk[sim_rsf_keep][sim_true_labels[sim_rsf_keep] == 0],
      curve = FALSE
    )
    
    sim_results_store[[profile_name]]$auprc_rsf[i] <- sim_rsf_pr$auc.integral
    sim_results_store[[profile_name]]$auc_rsf[i]   <- sim_rsf_auc
    
    ## ==========================================
    ## RSF (SMOTE)
    ## Same as RSF (No SMOTE) but trained on
    ## SMOTE-balanced data; time values resampled
    ## from empirical class-specific distributions
    ## ==========================================
    
    sim_rsf_smote_x <- sim_rsf_train %>% dplyr::select(-status, -time)
    sim_rsf_smote_y <- sim_rsf_train$status
    
    sim_rsf_smote_k <- min(5, max(1, min(table(sim_rsf_smote_y)) - 1))
    
    sim_rsf_smote_out <- SMOTE(
      X        = sim_rsf_smote_x,
      target   = sim_rsf_smote_y,
      K        = sim_rsf_smote_k,
      dup_size = 1
    )
    
    sim_rsf_smote_bal         <- sim_rsf_smote_out$data
    sim_rsf_smote_bal$status  <- as.numeric(as.character(sim_rsf_smote_bal$class))
    sim_rsf_smote_bal$class   <- NULL
    
    # Reattach empirical time values by class
    sim_rsf_smote_bal$time <- NA_real_
    idx0 <- which(sim_rsf_smote_bal$status == 0)
    idx1 <- which(sim_rsf_smote_bal$status == 1)
    
    sim_rsf_smote_bal$time[idx0] <- sample(
      sim_rsf_train$time[sim_rsf_train$status == 0],
      length(idx0), replace = TRUE
    )
    sim_rsf_smote_bal$time[idx1] <- sample(
      sim_rsf_train$time[sim_rsf_train$status == 1],
      length(idx1), replace = TRUE
    )
    
    sim_rsf_smote_model <- rfsrc(
      Surv(time, status) ~ .,
      data      = sim_rsf_smote_bal,
      ntree     = 100,
      nodesize  = 15,
      splitrule = "logrank",
      nthread   = 4,
      fast      = TRUE
    )
    
    sim_rsf_smote_pred <- predict(
      sim_rsf_smote_model,
      newdata = sim_rsf_test[, sim_rsf_smote_model$xvar.names, drop = FALSE]
    )
    sim_rsf_smote_risk <- as.numeric(
      1 - sim_rsf_smote_pred$survival[, ncol(sim_rsf_smote_pred$survival)]
    )
    
    sim_rsf_smote_auc  <- as.numeric(auc(roc(
      response  = sim_true_labels,
      predictor = sim_rsf_smote_risk,
      quiet     = TRUE
    )))
    
    sim_rsf_smote_keep <- is.finite(sim_rsf_smote_risk) & is.finite(sim_true_labels)
    sim_rsf_smote_pr   <- pr.curve(
      scores.class0 = sim_rsf_smote_risk[sim_rsf_smote_keep][sim_true_labels[sim_rsf_smote_keep] == 1],
      scores.class1 = sim_rsf_smote_risk[sim_rsf_smote_keep][sim_true_labels[sim_rsf_smote_keep] == 0],
      curve = FALSE
    )
    
    sim_results_store[[profile_name]]$auprc_rsf_smote[i] <- sim_rsf_smote_pr$auc.integral
    sim_results_store[[profile_name]]$auc_rsf_smote[i]   <- sim_rsf_smote_auc
    
    # Clean up large objects to free memory
    rm(sim_rsf_model, sim_rsf_pred,
       sim_rsf_smote_model, sim_rsf_smote_pred,
       sim_rsf_smote_bal, sim_rsf_smote_out)
    gc()
  }
}

## ----------------------------------------------------------
## 4. Compile and Save Results
## ----------------------------------------------------------

sim_results_compiled <- list()

for (profile_name in names(amplitude_profiles)) {
  sim_results_compiled[[profile_name]] <- data.frame(
    Profile            = profile_name,
    Amplitude          = amplitude_profiles[[profile_name]],
    AUPRC_LR           = mean(sim_results_store[[profile_name]]$auprc_lr),
    AUPRC_LR_SD        = sd(sim_results_store[[profile_name]]$auprc_lr),
    AUPRC_RSF          = mean(sim_results_store[[profile_name]]$auprc_rsf),
    AUPRC_RSF_SD       = sd(sim_results_store[[profile_name]]$auprc_rsf),
    AUPRC_RSF_SMOTE    = mean(sim_results_store[[profile_name]]$auprc_rsf_smote),
    AUPRC_RSF_SMOTE_SD = sd(sim_results_store[[profile_name]]$auprc_rsf_smote),
    AUC_LR             = mean(sim_results_store[[profile_name]]$auc_lr),
    AUC_LR_SD          = sd(sim_results_store[[profile_name]]$auc_lr),
    AUC_RSF            = mean(sim_results_store[[profile_name]]$auc_rsf),
    AUC_RSF_SD         = sd(sim_results_store[[profile_name]]$auc_rsf),
    AUC_RSF_SMOTE      = mean(sim_results_store[[profile_name]]$auc_rsf_smote),
    AUC_RSF_SMOTE_SD   = sd(sim_results_store[[profile_name]]$auc_rsf_smote)
  )
}

results_sim_nhpp <- do.call(rbind, sim_results_compiled)
results_sim_nhpp$Profile <- factor(
  results_sim_nhpp$Profile,
  levels = c("none", "mild", "moderate", "strong")
)

saveRDS(results_sim_nhpp, "sim_nhpp_results.rds")
readRDS("sim_nhpp_results.rds")

pander(
  results_sim_nhpp %>%
    dplyr::select(Profile, Amplitude,
                  AUPRC_LR, AUPRC_LR_SD,
                  AUPRC_RSF, AUPRC_RSF_SD,
                  AUPRC_RSF_SMOTE, AUPRC_RSF_SMOTE_SD),
  digits  = 4,
  caption = "NHPP Simulation: Mean AUPRC by Amplitude Profile and Model"
)

## ----------------------------------------------------------
## 5. AUPRC Plot (primary)
## ----------------------------------------------------------

sim_plot_data <- results_sim_nhpp %>%
  pivot_longer(
    cols      = c(AUPRC_LR, AUPRC_RSF, AUPRC_RSF_SMOTE),
    names_to  = "Model",
    values_to = "Mean_AUPRC"
  ) %>%
  mutate(
    SD = case_when(
      Model == "AUPRC_LR"        ~ AUPRC_LR_SD,
      Model == "AUPRC_RSF"       ~ AUPRC_RSF_SD,
      Model == "AUPRC_RSF_SMOTE" ~ AUPRC_RSF_SMOTE_SD
    ),
    Model = dplyr::recode(Model,
                          "AUPRC_LR"        = "Logistic Regression",
                          "AUPRC_RSF"       = "RSF (No SMOTE)",
                          "AUPRC_RSF_SMOTE" = "RSF (SMOTE)"
    )
  )

ggplot(sim_plot_data,
       aes(x = Profile, y = Mean_AUPRC, colour = Model, group = Model)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = Mean_AUPRC - SD, ymax = Mean_AUPRC + SD),
    width = 0.15
  ) +
  scale_colour_manual(values = c(
    "Logistic Regression" = "cornflowerblue",
    "RSF (No SMOTE)"      = "darkorange",
    "RSF (SMOTE)"         = "firebrick"
  )) +
  labs(
    title    = "AUPRC by Daily Cycle Amplitude (NHPP Simulation)",
    subtitle = bquote(lambda[i](t) == lambda[0](t) %.% exp(x[i]^T * beta) ~
                        "— amplitude a varied across" ~ .(n_splits) ~
                        "splits, N =" ~ .(format(n, big.mark = ","))),
    x      = "Amplitude Profile (a = 0, 0.3, 0.6, 0.8)",
    y      = "Mean AUPRC (+/- 1 SD)",
    colour = "Model"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

## ----------------------------------------------------------
## 6. AUC Plot (supplementary)
## ----------------------------------------------------------

sim_auc_data <- results_sim_nhpp %>%
  pivot_longer(
    cols      = c(AUC_LR, AUC_RSF, AUC_RSF_SMOTE),
    names_to  = "Model",
    values_to = "Mean_AUC"
  ) %>%
  mutate(
    SD = case_when(
      Model == "AUC_LR"        ~ AUC_LR_SD,
      Model == "AUC_RSF"       ~ AUC_RSF_SD,
      Model == "AUC_RSF_SMOTE" ~ AUC_RSF_SMOTE_SD
    ),
    Model = dplyr::recode(Model,
                          "AUC_LR"        = "Logistic Regression",
                          "AUC_RSF"       = "RSF (No SMOTE)",
                          "AUC_RSF_SMOTE" = "RSF (SMOTE)"
    )
  )

ggplot(sim_auc_data,
       aes(x = Profile, y = Mean_AUC, colour = Model, group = Model)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = Mean_AUC - SD, ymax = Mean_AUC + SD),
    width = 0.15
  ) +
  scale_colour_manual(values = c(
    "Logistic Regression" = "cornflowerblue",
    "RSF (No SMOTE)"      = "darkorange",
    "RSF (SMOTE)"         = "firebrick"
  )) +
  labs(
    title    = "AUC by Daily Cycle Amplitude (NHPP Simulation)",
    subtitle = bquote(lambda[i](t) == lambda[0](t) %.% exp(x[i]^T * beta) ~
                        "— amplitude a varied across" ~ .(n_splits) ~
                        "splits, N =" ~ .(format(n, big.mark = ","))),
    x      = "Amplitude Profile (a = 0, 0.3, 0.6, 0.8)",
    y      = "Mean AUC (+/- 1 SD)",
    colour = "Model"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

message("=== NHPP Simulation complete ===")
message("Results saved to: sim_nhpp_results.rds")
