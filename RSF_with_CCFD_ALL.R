###############################################################################
####################### 1. Setup and Data Loading #############################
###############################################################################
## Step 1: Set seed
set.seed(44940076)

## Step 2: Set Working Directory
setwd("C:/Users/ashak/OneDrive/Knowledge/Research/5. Research Topic Experimentation")

## Step 3: Load Data
credit <- read.csv('creditcard.csv', header = TRUE)

## Step 4: Remove Missing Values
credit <- na.omit(credit)

###############################################################################
################ 2. Preprocessing and Imbalance Handling ######################
###############################################################################
## Step 1: Create Survival Variables
credit$time <- 1:nrow(credit)               # Proxy for time-to-event
credit$status <- credit$Class               # Event indicator (1 = fraud)

# Step 2: Downsample majority class (non-fraud) for memory efficiency
set.seed(44940076)
fraud <- subset(credit, Class == 1)
nonfraud <- subset(credit, Class == 0)
nonfraud_sample <- nonfraud[sample(nrow(nonfraud), 50000), ]

# Step 3: Combine all fraud cases with sampled non-fraud cases
credit_subset <- rbind(fraud, nonfraud_sample)



###############################################################################
###################### 3. Create the 'Credit' sample ##########################
###############################################################################
library(dplyr)

## Step 1: Drop the original 'Class' column
credit_filtered <- credit_subset %>%
  dplyr::select(-Class) %>%
  mutate(
    time = as.numeric(time),
    status = as.numeric(status)
  ) %>%
  na.omit()

# Convert character columns to factors
credit_filtered[] <- lapply(credit_filtered, function(x) {
  if (is.character(x)) as.factor(x) else x
})

# Optional: Sample a subset for benchmarking
set.seed(44940076)
credit_sample <- credit_filtered[sample(nrow(credit_filtered), 50000), ]

###############################################################################
##################### 4. Logistic Regression (No SMOTE) #######################
###############################################################################
library(caret)
library(pROC)
library(PRROC)

set.seed(44940076)

n_splits <- 30

# Storage vectors
sens_vec  <- numeric(n_splits)
prec_vec  <- numeric(n_splits)
auc_vec   <- numeric(n_splits)
auprc_vec <- numeric(n_splits)

for (i in 1:n_splits) {
  
  # Stratified 70–30 split
  train_idx <- createDataPartition(credit_sample$status, p = 0.7, list = FALSE)
  train_data <- credit_sample[train_idx, ]
  test_data  <- credit_sample[-train_idx, ]
  
  # Fit Logistic Regression
  logit_model <- glm(status ~ ., data = train_data, family = binomial)
  
  # Predict probabilities
  test_probs <- predict(logit_model, newdata = test_data, type = "response")
  
  # Convert to class labels (threshold 0.5)
  test_pred <- ifelse(test_probs > 0.5, 1, 0)
  
  # Confusion Matrix
  conf_matrix <- confusionMatrix(
    factor(test_pred),
    factor(test_data$status),
    positive = "1"
  )
  
  sens_vec[i] <- conf_matrix$byClass["Sensitivity"]
  prec_vec[i] <- conf_matrix$byClass["Precision"]
  
  # ROC + AUC
  true_labels <- as.numeric(as.character(test_data$status))
  roc_obj <- roc(response = true_labels, predictor = test_probs, quiet = TRUE)
  auc_vec[i] <- as.numeric(auc(roc_obj))
  
  # AUPRC
  pr_obj <- pr.curve(
    scores.class0 = test_probs[true_labels == 1],
    scores.class1 = test_probs[true_labels == 0],
    curve = FALSE
  )
  
  auprc_vec[i] <- pr_obj$auc.integral
}

# Results
results_lr_no_smote <- data.frame(
  Metric = c("Sensitivity", "Precision", "AUC", "AUPRC"),
  Mean = c(mean(sens_vec),
           mean(prec_vec),
           mean(auc_vec),
           mean(auprc_vec)),
  SD = c(sd(sens_vec),
         sd(prec_vec),
         sd(auc_vec),
         sd(auprc_vec))
)

print(results_lr_no_smote)

###############################################################################
####################### 5. Logistic Regression (SMOTE) ########################
###############################################################################
library(caret)
library(smotefamily)
library(dplyr)
library(pROC)
library(PRROC)

set.seed(44940076)

n_splits <- 30

sens_vec  <- numeric(n_splits)
prec_vec  <- numeric(n_splits)
auc_vec   <- numeric(n_splits)
auprc_vec <- numeric(n_splits)

for (i in 1:n_splits) {
  
  # Stratified 70–30 split (BEFORE SMOTE)
  train_idx <- createDataPartition(credit_sample$status, p = 0.7, list = FALSE)
  train_data <- credit_sample[train_idx, ]
  test_data  <- credit_sample[-train_idx, ]
  
  # Ensure numeric 0/1 labels
  train_data$status <- as.numeric(as.character(train_data$status))
  test_data$status  <- as.numeric(as.character(test_data$status))
  
  # Apply SMOTE to TRAINING ONLY 
  # Drop time from SMOTE input (keep deployment realistic)
  train_for_smote <- train_data %>% dplyr::select(-time)
  
  X_train <- train_for_smote %>% dplyr::select(-status)
  y_train <- train_for_smote$status
  
  # Safety: SMOTE K must be < minority count
  n_min <- min(table(y_train))
  K_use <- min(5, max(1, n_min - 1))
  
  smote_out <- SMOTE(
    X = X_train,
    target = y_train,
    K = K_use,
    dup_size = 3
  )
  
  train_smote <- smote_out$data
  train_smote$status <- as.numeric(as.character(train_smote$class))
  train_smote$class <- NULL
  
  # Fit Logistic Regression on SMOTE training
  logit_smote <- glm(status ~ ., data = train_smote, family = binomial)
  
  # Predict on untouched TEST data
  test_probs <- predict(logit_smote, newdata = test_data, type = "response")
  test_pred  <- ifelse(test_probs > 0.5, 1, 0)
  
  # Confusion matrix 
  cm <- confusionMatrix(
    factor(test_pred),
    factor(test_data$status),
    positive = "1"
  )
  
  sens_vec[i] <- cm$byClass["Sensitivity"]
  prec_vec[i] <- cm$byClass["Precision"]
  
  # ROC/AUC
  roc_obj <- roc(response = test_data$status, predictor = test_probs, quiet = TRUE)
  auc_vec[i] <- as.numeric(auc(roc_obj))
  
  # AUPRC
  true_labels <- test_data$status  # aligned by construction
  
  # PRROC needs scores split by class - ensure finite
  keep <- is.finite(test_probs) & is.finite(true_labels)
  scores <- test_probs[keep]
  labels <- true_labels[keep]
  
  pr_obj <- pr.curve(
    scores.class0 = scores[labels == 1],
    scores.class1 = scores[labels == 0],
    curve = FALSE
  )
  
  auprc_vec[i] <- pr_obj$auc.integral
}

# Results
results_lr_smote <- data.frame(
  Metric = c("Sensitivity", "Precision", "AUC", "AUPRC"),
  Mean = c(mean(sens_vec), mean(prec_vec), mean(auc_vec), mean(auprc_vec)),
  SD   = c(sd(sens_vec),   sd(prec_vec),   sd(auc_vec),   sd(auprc_vec))
)

print(results_lr_smote)

###############################################################################
##################### 6. Random Survival Forest (no SMOTE) ####################
###############################################################################
library(caret)
library(randomForestSRC)
library(pROC)
library(PRROC)

set.seed(44940076)

n_splits <- 30

sens_vec  <- numeric(n_splits)
prec_vec  <- numeric(n_splits)
auc_vec   <- numeric(n_splits)
auprc_vec <- numeric(n_splits)

for (i in 1:n_splits) {
  
  # Stratified 70–30 split
  train_idx <- createDataPartition(credit_sample$status, p = 0.7, list = FALSE)
  train_data <- credit_sample[train_idx, ]
  test_data  <- credit_sample[-train_idx, ]
  
  # Ensure numeric 0/1
  train_data$status <- as.numeric(as.character(train_data$status))
  test_data$status  <- as.numeric(as.character(test_data$status))
  
  # Fit RSF (no SMOTE)
  rsf_model <- rfsrc(
    Surv(time, status) ~ .,
    data = train_data,
    ntree = 100,
    nodesize = 15,
    splitrule = "logrank",
    nthread = 1,
    fast = TRUE
  )
  
  # Predict (ensure newdata has only training predictors)
  test_x <- test_data[, rsf_model$xvar.names, drop = FALSE]
  rsf_pred <- predict(rsf_model, newdata = test_x)
  
  # Risk score: 1 - S(tmax)
  risk <- 1 - rsf_pred$survival[, ncol(rsf_pred$survival)]
  risk <- as.numeric(risk)
  
  stopifnot(length(risk) == nrow(test_data))
  
  # Classification (median threshold)
  threshold <- median(risk)
  pred_class <- ifelse(risk > threshold, 1, 0)
  
  # Confusion matrix
  cm <- confusionMatrix(
    factor(pred_class),
    factor(test_data$status),
    positive = "1"
  )
  
  sens_vec[i] <- cm$byClass["Sensitivity"]
  prec_vec[i] <- cm$byClass["Precision"]
  
  # ROC / AUC
  roc_obj <- roc(
    response = test_data$status,
    predictor = risk,
    quiet = TRUE
  )
  auc_vec[i] <- as.numeric(auc(roc_obj))
  
  # AUPRC
  true_labels <- test_data$status
  
  keep <- is.finite(risk) & is.finite(true_labels)
  scores <- risk[keep]
  labels <- true_labels[keep]
  
  pr_obj <- pr.curve(
    scores.class0 = scores[labels == 1],
    scores.class1 = scores[labels == 0],
    curve = FALSE
  )
  
  auprc_vec[i] <- pr_obj$auc.integral
}

# Results
results_rsf_no_smote <- data.frame(
  Metric = c("Sensitivity", "Precision", "AUC", "AUPRC"),
  Mean = c(mean(sens_vec), mean(prec_vec), mean(auc_vec), mean(auprc_vec)),
  SD   = c(sd(sens_vec),   sd(prec_vec),   sd(auc_vec),   sd(auprc_vec))
)

print(results_rsf_no_smote)


###############################################################################
###################### 7. Random Survival Forest (SMOTE) ######################
###############################################################################
library(caret)
library(smotefamily)
library(dplyr)
library(randomForestSRC)
library(pROC)
library(PRROC)

set.seed(44940076)

n_splits <- 30

sens_vec  <- numeric(n_splits)
prec_vec  <- numeric(n_splits)
auc_vec   <- numeric(n_splits)
auprc_vec <- numeric(n_splits)

for (i in 1:n_splits) {
  
  # Stratified 70–30 split BEFORE SMOTE
  train_idx <- createDataPartition(credit_sample$status, p = 0.7, list = FALSE)
  train_data <- credit_sample[train_idx, ]
  test_data  <- credit_sample[-train_idx, ]
  
  # Ensure numeric 0/1
  train_data$status <- as.numeric(as.character(train_data$status))
  test_data$status  <- as.numeric(as.character(test_data$status))
  
  # SMOTE on training predictors only (exclude time)
  train_X <- train_data %>% dplyr::select(-status, -time)
  train_y <- train_data$status
  
  # Safety: choose K based on minority count in this split
  n_min <- min(table(train_y))
  K_use <- min(5, max(1, n_min - 1))
  
  smote_out <- SMOTE(
    X = train_X,
    target = train_y,
    K = K_use,
    dup_size = 3
  )
  
  train_smote <- smote_out$data
  train_smote$status <- as.numeric(as.character(train_smote$class))
  train_smote$class <- NULL
  
  # Reattach empirical time (limitation you already state)
  train_smote$time <- NA_real_
  
  idx0 <- which(train_smote$status == 0)
  idx1 <- which(train_smote$status == 1)
  
  train_smote$time[idx0] <- sample(train_data$time[train_data$status == 0],
                                   length(idx0), replace = TRUE)
  train_smote$time[idx1] <- sample(train_data$time[train_data$status == 1],
                                   length(idx1), replace = TRUE)
  
  # Fit RSF on SMOTE training
  rsf_model <- rfsrc(
    Surv(time, status) ~ .,
    data = train_smote,
    ntree = 100,
    nodesize = 15,
    splitrule = "logrank",
    nthread = 1,
    fast = TRUE
  )
  
  # Predict on untouched test data
  # Ensure newdata has only the predictors RSF expects
  test_x <- test_data[, rsf_model$xvar.names, drop = FALSE]
  
  rsf_pred <- predict(rsf_model, newdata = test_x)
  
  # Risk score: 1 - S(tmax)
  risk <- 1 - rsf_pred$survival[, ncol(rsf_pred$survival)]
  risk <- as.numeric(risk)
  
  stopifnot(length(risk) == nrow(test_data))
  
  # Classification threshold (median risk)
  threshold <- median(risk)
  pred_class <- ifelse(risk > threshold, 1, 0)
  
  cm <- confusionMatrix(
    factor(pred_class),
    factor(test_data$status),
    positive = "1"
  )
  
  sens_vec[i] <- cm$byClass["Sensitivity"]
  prec_vec[i] <- cm$byClass["Precision"]
  
  # ROC / AUC
  roc_obj <- roc(response = test_data$status, predictor = risk, quiet = TRUE)
  auc_vec[i] <- as.numeric(auc(roc_obj))
  
  # AUPRC
  true_labels <- test_data$status
  
  keep <- is.finite(risk) & is.finite(true_labels)
  scores <- risk[keep]
  labels <- true_labels[keep]
  
  pr_obj <- pr.curve(
    scores.class0 = scores[labels == 1],
    scores.class1 = scores[labels == 0],
    curve = FALSE
  )
  
  auprc_vec[i] <- pr_obj$auc.integral
}

# Results
results_rsf_smote <- data.frame(
  Metric = c("Sensitivity", "Precision", "AUC", "AUPRC"),
  Mean = c(mean(sens_vec), mean(prec_vec), mean(auc_vec), mean(auprc_vec)),
  SD   = c(sd(sens_vec),   sd(prec_vec),   sd(auc_vec),   sd(auprc_vec))
)

print(results_rsf_smote)

###############################################################################
######################### 8. AUPRC (with Mean and SD) #########################
###############################################################################
df_sum <- data.frame(
  Model = c("RSF (SMOTE)", "RSF (No)", "LogReg (SMOTE)", "LogReg (No)"),
  AUPRC  = c(0.8720122, 0.8510782, 0.8597711, 0.8624825),
  SD    = c(0.0278027, 0.0301426, 0.0312189, 0.0235565)
)

# Order top to bottom
df_sum$Model <- factor(
  df_sum$Model,
  levels = c("RSF (SMOTE)", "LogReg (No)", "LogReg (SMOTE)", "RSF (No)")
)

# Highlight best
library(ggplot2)

df_sum$Highlight <- ifelse(df_sum$Model == "RSF (SMOTE)", "Best", "Other")

ggplot(df_sum, aes(x = Model, y = AUPRC, color = Highlight)) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = AUPRC - SD, ymax = AUPRC + SD),
                width = 0.15) +
  scale_color_manual(values = c("Best"="firebrick2", "Other"="grey30"))  +
  coord_flip() +
  labs(
    x = "Model",
    y = "AUPRC",
    title = "Ranking performance under extreme class imbalance"
  ) +
  theme_bw(base_size = 14) +
  theme(
    # axis text and ticks black
    axis.text = element_text(color = "black"),
    axis.title = element_text(color = "black"),
    axis.ticks = element_line(color = "black"),
    
    # grid styling
    panel.grid.major = element_line(color = "grey95"),
    panel.grid.minor = element_line(color = "grey95"),
    
    # remove legend (since obvious)
    legend.position = "none"
  )

###############################################################################
###################### 9. Sensitivity (with Mean and SD) ######################
###############################################################################

df_sens <- data.frame(
  Model = c("RSF (SMOTE)", "RSF (No)", "LogReg (SMOTE)", "LogReg (No)"),
  Sensitivity = c(0.9875298, 0.9875532, 0.8350287, 0.7913633)
)

df_sens$Model <- factor(df_sens$Model,
                        levels = c("RSF (SMOTE)", "RSF (No)", "LogReg (SMOTE)", "LogReg (No)"))

df_sens$Highlight <- ifelse(df_sens$Model == "RSF (SMOTE)", "Best", "Other")

ggplot(df_sens, aes(x = Model, y = Sensitivity, fill = Highlight)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", Sensitivity)),
            hjust = -0.1, size = 4, color = "black") +
  coord_flip() +
  scale_fill_manual(values = c("Best"="firebrick3", "Other"="grey75")) +
  scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.2)) +
  labs(
    title = "Sensitivity (Proactive Fraud Detection)",
    x = "Model",
    y = "Sensitivity (Recall)"
  ) +
  theme_bw(base_size = 14) +
  theme(
    legend.position = "none",
    axis.text = element_text(color = "black"),
    axis.title = element_text(color = "black")
  )

###############################################################################
######################## 10. SHAP for RSF with SMOTE (TO FIX) ###########################
###############################################################################

# Step 1: Install and load iml
# install.packages("iml")
library(iml)

# Predictor object
predict_function <- function(model, newdata) {
  p <- predict(model, newdata = newdata)
  risk <- 1 - p$survival[, ncol(p$survival)]
  as.numeric(risk)
}

# Pick one observation for Shapley
X_train <- train_data_smote %>% dplyr::select(-status, -time)
y_train <- train_data_smote$status

predictor_rsf <- Predictor$new(
  model = rsf_model_smote,
  data = X_train,
  y = y_train,
  predict.function = predict_function
)

x_interest <- test_data %>%
  dplyr::select(colnames(X_train)) %>%
  slice(1)

shapley <- Shapley$new(predictor_rsf, x.interest = x_interest)
plot(shapley) +
  scale_fill_manual(values = "#1f77b4")

p <- plot(shapley)
p$layers[[1]]$aes_params$fill <- "#1f77b4"
p

###############################################################################
######################## 11. LIME for RSF with SMOTE (TO FIX) ###########################
###############################################################################
# Step 1: Load packages
library(dplyr)
library(lime)

# Step 2: Establish the features used by the model (excluding the outcome and Time)
X_test <- test_data %>% dplyr::select(colnames(X_train))

# Step 3: RSF risk score function (same logic as SHAP)
rsf_risk <- function(model, newdata) {
  p <- predict(model, newdata = newdata)
  risk <- 1 - p$survival[, ncol(p$survival)]
  as.numeric(risk)
}

# Step 4: Calibrate the risk (probability using a simple logistic calibration).
train_risk <- rsf_risk(rsf_model_smote, X_train)

cal_model <- glm(y_train ~ train_risk, family = binomial)

predict_proba_rsf <- function(newdata) {
  r <- rsf_risk(rsf_model_smote, newdata)
  p1 <- as.numeric(predict(cal_model, newdata = data.frame(train_risk = r), type = "response"))
  # return a 2-column data.frame of class: "0" and "1"
  data.frame("0" = 1 - p1, "1" = p1)
}

# Step 5: Ensure LIME creates this in a tabular manner.
explainer <- lime(
  x = X_train,
  model = rsf_model_smote,
  bin_continuous = TRUE
)

# Step 6: Create Explanations.
set.seed(44940076)
x_interest <- X_test %>% slice(1)

model_type.rfsrc <- function(x, ...) "classification"

predict_model.rfsrc <- function(x, newdata, type, ...) {
  # ignore type; return class probabilities with columns "0" and "1"
  predict_proba_rsf(newdata)
}

# Step 7: Make it 'Explain' the model.
explanation <- explain(
  x = x_interest,
  explainer = explainer,
  n_features = 8,
  n_labels = 1   # explain the top predicted label
)

# Step 8: Plot the model.
plot_features(explanation)