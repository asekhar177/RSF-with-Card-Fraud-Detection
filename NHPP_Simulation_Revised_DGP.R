## ============================================================
## Revised Simulation: Synthetic DGP with Known Temporal Structure
## ============================================================
## Purpose: Test whether RSF gains a discriminatory advantage over
## Logistic Regression when genuine time-to-event structure exists
## alongside informative features.
##
## Design: Generate synthetic data where fraud risk depends on BOTH
## a feature-based linear predictor AND a time-dependent baseline
## hazard. The temporal component's strength is varied across four
## profiles (none, mild, moderate, strong) while the feature signal
## remains constant. If RSF can exploit temporal structure, its
## AUPRC advantage should grow as temporal intensity increases.
## ============================================================

library(caret)
library(dplyr)
library(smotefamily)
library(randomForestSRC)
library(pROC)
library(PRROC)
library(ggplot2)
library(tidyr)

set.seed(44940076)

## ----------------------------------------------------------
## 1. Simulation Parameters
## ----------------------------------------------------------

N           <- 50000L     # Total observations (matches downsampled ULB size)
n_fraud     <- 492L       # Match ULB fraud count for realistic prevalence
n_features  <- 10L        # Informative features (keep manageable)
n_noise     <- 5L         # Noise features (no predictive value)
n_splits    <- 10L        # Repeated splits per profile
beta_true   <- c(1.2, -0.8, 0.6, -0.5, 0.9,   # True coefficients for informative features
                 0.4, -0.3, 0.7, -0.6, 0.5)

## ----------------------------------------------------------
## 2. Define Temporal Intensity Profiles
## ----------------------------------------------------------
## Each function takes normalised time (0-1) and returns a
## multiplicative hazard modifier. Higher values = more fraud risk
## at that time point.

temporal_profiles <- list(
  none = list(
    fn = function(t) rep(1, length(t)),
    label = "No temporal structure"
  ),
  mild = list(
    fn = function(t) 1 + 0.5 * sin(2 * pi * t),
    label = "Mild sinusoidal"
  ),
  moderate = list(
    fn = function(t) {
      1 + 2.0 * (exp(-((t - 0.15)^2) / 0.005) +
                   exp(-((t - 0.65)^2) / 0.005))
    },
    label = "Moderate peaked"
  ),
  strong = list(
    fn = function(t) {
      1 + 5.0 * (exp(-((t - 0.15)^2) / 0.003) +
                   exp(-((t - 0.65)^2) / 0.003))
    },
    label = "Strong peaked"
  )
)

## ----------------------------------------------------------
## 3. Data Generation Function
## ----------------------------------------------------------
## Generates a dataset where fraud probability depends on:
##   P(fraud | X, t) ∝ h0(t) * exp(beta' X)
##
## - h0(t) is the temporal profile (baseline hazard modifier)
## - beta'X is the feature-based linear predictor
## - Labels are assigned by sampling n_fraud cases with probability
##   proportional to the combined risk score
##
## This preserves the feature-label association while adding
## controlled temporal structure.

generate_sim_data <- function(N, n_fraud, n_features, n_noise,
                               beta_true, temporal_fn, seed = 42) {
  set.seed(seed)

  ## Generate features: informative + noise
  X_informative <- matrix(rnorm(N * n_features), nrow = N, ncol = n_features)
  colnames(X_informative) <- paste0("X", 1:n_features)

  X_noise <- matrix(rnorm(N * n_noise), nrow = N, ncol = n_noise)
  colnames(X_noise) <- paste0("noise", 1:n_noise)

  ## Generate time variable (uniform over observation window)
  Time <- runif(N, min = 0, max = 172800)  # 48 hours in seconds, matching ULB

  ## Compute linear predictor from features
  lp <- as.numeric(X_informative %*% beta_true)

  ## Compute temporal modifier
  t_norm <- (Time - min(Time)) / (max(Time) - min(Time))
  h0_t <- temporal_fn(t_norm)

  ## Combined risk score: feature signal * temporal modifier
  risk_score <- exp(lp) * h0_t

  ## Normalise to probabilities and sample fraud labels
  risk_prob <- risk_score / sum(risk_score)
  fraud_idx <- sample(N, size = n_fraud, replace = FALSE, prob = risk_prob)

  ## Construct dataset
  sim_df <- data.frame(X_informative, X_noise)
  sim_df$Time <- Time
  sim_df$time <- 1:N  # Proxy time index for RSF (row ordering)
  sim_df$status <- 0L
  sim_df$status[fraud_idx] <- 1L

  sim_df
}

## ----------------------------------------------------------
## 4. Model Evaluation Loop
## ----------------------------------------------------------

message("=== Starting revised simulation ===")
message("Profiles: ", paste(names(temporal_profiles), collapse = ", "))
message("Splits per profile: ", n_splits)
message("")

sim_results_store <- list()

for (profile_name in names(temporal_profiles)) {
  message("--- Profile: ", profile_name, " (", 
          temporal_profiles[[profile_name]]$label, ") ---")

  sim_results_store[[profile_name]] <- list(
    auprc_lr       = numeric(n_splits),
    auprc_rsf      = numeric(n_splits),
    auprc_rsf_smote = numeric(n_splits),
    auc_lr         = numeric(n_splits),
    auc_rsf        = numeric(n_splits),
    auc_rsf_smote  = numeric(n_splits)
  )

  ## Generate one dataset per profile (same across splits for consistency)
  sim_data <- generate_sim_data(
    N = N, n_fraud = n_fraud, n_features = n_features,
    n_noise = n_noise, beta_true = beta_true,
    temporal_fn = temporal_profiles[[profile_name]]$fn,
    seed = 44940076
  )

  for (i in seq_len(n_splits)) {
    message("  Split ", i, " of ", n_splits, " - ", Sys.time())

    ## 1. Stratified 70-30 split
    sim_train_idx <- createDataPartition(sim_data$status, p = 0.7, list = FALSE)
    sim_train <- sim_data[sim_train_idx, ]
    sim_test  <- sim_data[-sim_train_idx, ]

    sim_train$status <- as.numeric(as.character(sim_train$status))
    sim_test$status  <- as.numeric(as.character(sim_test$status))
    sim_true_labels  <- sim_test$status

    ## ==========================================
    ## LOGISTIC REGRESSION (No SMOTE)
    ## ==========================================

    ## Feature columns only (exclude time, proxy time, status)
    feature_cols <- grep("^(X|noise)", names(sim_train), value = TRUE)

    sim_lr_formula <- as.formula(
      paste("status ~", paste(feature_cols, collapse = " + "), "+ Time")
    )
    sim_lr_model <- glm(sim_lr_formula, data = sim_train, family = binomial)
    sim_lr_prob  <- predict(sim_lr_model, newdata = sim_test, type = "response")

    ## AUC
    sim_lr_auc <- as.numeric(auc(roc(
      response = sim_true_labels, predictor = sim_lr_prob, quiet = TRUE
    )))

    ## AUPRC
    sim_lr_keep <- is.finite(sim_lr_prob) & is.finite(sim_true_labels)
    sim_lr_pr <- pr.curve(
      scores.class0 = sim_lr_prob[sim_lr_keep][sim_true_labels[sim_lr_keep] == 1],
      scores.class1 = sim_lr_prob[sim_lr_keep][sim_true_labels[sim_lr_keep] == 0],
      curve = FALSE
    )

    sim_results_store[[profile_name]]$auprc_lr[i] <- sim_lr_pr$auc.integral
    sim_results_store[[profile_name]]$auc_lr[i]   <- sim_lr_auc

    ## ==========================================
    ## RSF (No SMOTE)
    ## ==========================================

    sim_rsf_train <- sim_train %>% dplyr::select(-Time)
    sim_rsf_test  <- sim_test  %>% dplyr::select(-Time)

    sim_rsf_model <- rfsrc(
      Surv(time, status) ~ .,
      data     = sim_rsf_train,
      ntree    = 100,
      nodesize = 15,
      splitrule = "logrank",
      nthread  = 4,
      fast     = TRUE
    )

    sim_rsf_test_x <- sim_rsf_test[, sim_rsf_model$xvar.names, drop = FALSE]
    sim_rsf_pred   <- predict(sim_rsf_model, newdata = sim_rsf_test_x)
    sim_rsf_risk   <- 1 - sim_rsf_pred$survival[, ncol(sim_rsf_pred$survival)]
    sim_rsf_risk   <- as.numeric(sim_rsf_risk)

    ## AUC
    sim_rsf_auc <- as.numeric(auc(roc(
      response = sim_true_labels, predictor = sim_rsf_risk, quiet = TRUE
    )))

    ## AUPRC
    sim_rsf_keep <- is.finite(sim_rsf_risk) & is.finite(sim_true_labels)
    sim_rsf_pr <- pr.curve(
      scores.class0 = sim_rsf_risk[sim_rsf_keep][sim_true_labels[sim_rsf_keep] == 1],
      scores.class1 = sim_rsf_risk[sim_rsf_keep][sim_true_labels[sim_rsf_keep] == 0],
      curve = FALSE
    )

    sim_results_store[[profile_name]]$auprc_rsf[i] <- sim_rsf_pr$auc.integral
    sim_results_store[[profile_name]]$auc_rsf[i]   <- sim_rsf_auc

    ## ==========================================
    ## RSF (SMOTE)
    ## ==========================================

    sim_rsf_smote_x <- sim_rsf_train %>% dplyr::select(-status, -time)
    sim_rsf_smote_y <- sim_rsf_train$status

    sim_rsf_smote_nmin <- min(table(sim_rsf_smote_y))
    sim_rsf_smote_k    <- min(5, max(1, sim_rsf_smote_nmin - 1))

    sim_rsf_smote_out <- SMOTE(
      X      = sim_rsf_smote_x,
      target = sim_rsf_smote_y,
      K      = sim_rsf_smote_k,
      dup_size = 1
    )

    sim_rsf_smote_bal <- sim_rsf_smote_out$data
    sim_rsf_smote_bal$status <- as.numeric(as.character(sim_rsf_smote_bal$class))
    sim_rsf_smote_bal$class  <- NULL

    ## Reattach empirical time values
    sim_rsf_smote_bal$time <- NA_real_
    sim_rsf_smote_idx0 <- which(sim_rsf_smote_bal$status == 0)
    sim_rsf_smote_idx1 <- which(sim_rsf_smote_bal$status == 1)

    sim_rsf_smote_bal$time[sim_rsf_smote_idx0] <- sample(
      sim_rsf_train$time[sim_rsf_train$status == 0],
      length(sim_rsf_smote_idx0), replace = TRUE
    )
    sim_rsf_smote_bal$time[sim_rsf_smote_idx1] <- sample(
      sim_rsf_train$time[sim_rsf_train$status == 1],
      length(sim_rsf_smote_idx1), replace = TRUE
    )

    sim_rsf_smote_model <- rfsrc(
      Surv(time, status) ~ .,
      data     = sim_rsf_smote_bal,
      ntree    = 100,
      nodesize = 15,
      splitrule = "logrank",
      nthread  = 4,
      fast     = TRUE
    )

    sim_rsf_smote_test_x <- sim_rsf_test[, sim_rsf_smote_model$xvar.names, drop = FALSE]
    sim_rsf_smote_pred   <- predict(sim_rsf_smote_model, newdata = sim_rsf_smote_test_x)
    sim_rsf_smote_risk   <- 1 - sim_rsf_smote_pred$survival[, ncol(sim_rsf_smote_pred$survival)]
    sim_rsf_smote_risk   <- as.numeric(sim_rsf_smote_risk)

    ## AUC
    sim_rsf_smote_auc <- as.numeric(auc(roc(
      response = sim_true_labels, predictor = sim_rsf_smote_risk, quiet = TRUE
    )))

    ## AUPRC
    sim_rsf_smote_keep <- is.finite(sim_rsf_smote_risk) & is.finite(sim_true_labels)
    sim_rsf_smote_pr <- pr.curve(
      scores.class0 = sim_rsf_smote_risk[sim_rsf_smote_keep][sim_true_labels[sim_rsf_smote_keep] == 1],
      scores.class1 = sim_rsf_smote_risk[sim_rsf_smote_keep][sim_true_labels[sim_rsf_smote_keep] == 0],
      curve = FALSE
    )

    sim_results_store[[profile_name]]$auprc_rsf_smote[i] <- sim_rsf_smote_pr$auc.integral
    sim_results_store[[profile_name]]$auc_rsf_smote[i]   <- sim_rsf_smote_auc

    ## Clean up
    rm(sim_rsf_model, sim_rsf_pred, sim_rsf_smote_model,
       sim_rsf_smote_pred, sim_rsf_smote_bal, sim_rsf_smote_out)
    gc()
  }
}

## ----------------------------------------------------------
## 5. Compile and Save Results
## ----------------------------------------------------------

sim_results_compiled <- list()
for (profile_name in names(temporal_profiles)) {
  sim_results_compiled[[profile_name]] <- data.frame(
    Profile          = profile_name,
    AUPRC_LR         = mean(sim_results_store[[profile_name]]$auprc_lr),
    AUPRC_LR_SD      = sd(sim_results_store[[profile_name]]$auprc_lr),
    AUPRC_RSF        = mean(sim_results_store[[profile_name]]$auprc_rsf),
    AUPRC_RSF_SD     = sd(sim_results_store[[profile_name]]$auprc_rsf),
    AUPRC_RSF_SMOTE  = mean(sim_results_store[[profile_name]]$auprc_rsf_smote),
    AUPRC_RSF_SMOTE_SD = sd(sim_results_store[[profile_name]]$auprc_rsf_smote),
    AUC_LR           = mean(sim_results_store[[profile_name]]$auc_lr),
    AUC_LR_SD        = sd(sim_results_store[[profile_name]]$auc_lr),
    AUC_RSF          = mean(sim_results_store[[profile_name]]$auc_rsf),
    AUC_RSF_SD       = sd(sim_results_store[[profile_name]]$auc_rsf),
    AUC_RSF_SMOTE    = mean(sim_results_store[[profile_name]]$auc_rsf_smote),
    AUC_RSF_SMOTE_SD = sd(sim_results_store[[profile_name]]$auc_rsf_smote)
  )
}

results_sim_revised <- do.call(rbind, sim_results_compiled)
results_sim_revised$Profile <- factor(
  results_sim_revised$Profile,
  levels = c("none", "mild", "moderate", "strong")
)

saveRDS(results_sim_revised, "sim_revised_results.rds")

## Print summary table
library(pander)
pander(
  results_sim_revised %>%
    dplyr::select(Profile, AUPRC_LR, AUPRC_LR_SD, AUPRC_RSF, AUPRC_RSF_SD,
                  AUPRC_RSF_SMOTE, AUPRC_RSF_SMOTE_SD),
  digits = 4,
  caption = "Revised Simulation: Mean AUPRC by Temporal Profile and Model"
)

## ----------------------------------------------------------
## 6. Visualisation: AUPRC by Profile
## ----------------------------------------------------------
library(tidyverse)

sim_plot_data <- results_sim_revised %>%
  pivot_longer(
    cols = c(AUPRC_LR, AUPRC_RSF, AUPRC_RSF_SMOTE),
    names_to = "Model",
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
    title    = "AUPRC by Temporal Risk Profile (Revised Simulation)",
    subtitle = paste0("Synthetic Data Gen Profile with known feature + temporal structure (",
                      n_splits, " splits, N = ", format(N, big.mark = ","), ")"),
    x = "Temporal Intensity Profile",
    y = "Mean AUPRC (+/- 1 SD)",
    colour = "Model"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

## ----------------------------------------------------------
## 7. Visualisation: AUC by Profile (supplementary)
## ----------------------------------------------------------

sim_auc_plot_data <- results_sim_revised %>%
  pivot_longer(
    cols = c(AUC_LR, AUC_RSF, AUC_RSF_SMOTE),
    names_to = "Model",
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

ggplot(sim_auc_plot_data,
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
    title    = "AUC by Temporal Risk Profile (Revised Simulation)",
    subtitle = paste0("Synthetic Data Gen Profile with known feature + temporal structure (",
                      n_splits, " splits, N = ", format(N, big.mark = ","), ")"),
    x = "Temporal Intensity Profile",
    y = "Mean AUC (+/- 1 SD)",
    colour = "Model"
  ) +
  theme_bw() +
  theme(legend.position = "bottom")

message("=== Simulation complete ===")
message("Results saved to: sim_revised_results.rds")


# ==============================================================================
# Research Methodology Evaluation & Modelling Session
# Location: University of Western Australia (UWA), Reid Library Ground Floor
# Date: Wednesday, 20 May 2026
# Network Anchor: eduroam authentication via Macquarie University
# ==============================================================================
