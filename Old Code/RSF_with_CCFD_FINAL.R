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

# Step 1: Re-attach the target variable
credit_sample$status <- as.factor(credit_sample$status)  # Ensure it's a factor for classification

# Step 2: Train-Test Split
set.seed(44940076)
train_indices <- sample(seq_len(nrow(credit_sample)), size = 0.7 * nrow(credit_sample))
train_data <- credit_sample[train_indices, ]
test_data  <- credit_sample[-train_indices, ]

# Step 3: Fit Logistic Regression Model
logit_model <- glm(status ~ ., data = train_data, family = binomial)

# Step 4: Predict Probabilities on Test Set
test_probs <- predict(logit_model, newdata = test_data, type = "response")

# Step 5: Convert Probabilities to Class Labels (Threshold = 0.5)
test_pred <- ifelse(test_probs > 0.5, 1, 0) # Change the scale from 0.05 to a threshold of choice.

# Step 6: Evaluate Performance
library(caret)
conf_matrix <- confusionMatrix(factor(test_pred), factor(test_data$status), positive = "1")

# Step 7: Output Key Metrics
print(conf_matrix)

## Produce ROC
# Step 8: Load required package
#install.packages("pROC")
library(pROC)

# Step 9: Ensure true labels are numeric
true_labels <- as.numeric(as.character(test_data$status))


###############################################################################
# ROC + AUC
###############################################################################
# Step 10: Generate ROC
roc_logit <- roc(response = true_labels, predictor = test_probs)

# Step 11: Plot ROC curve
plot(roc_logit, col = "green", lwd = 2, main = "ROC Curve — Logistic Regression (No SMOTE)")
abline(a = 0, b = 1, lty = 2, col = "gray")

# Step 12: Print AUC
auc_logit <- auc(roc_logit)
print(paste("AUC (No SMOTE):", round(auc_logit, 4)))



###############################################################################
# AUPRC
###############################################################################
# Step 13: Use PRROC package. 
library(PRROC)

# Step 14: Ensure true labels are numeric (0-1).
true_labels <- as.numeric(as.character(test_data$status))

# Step 15: Calculate predicted probabilities from logit model.
scores <- test_probs  

# Step 16: Compute PR curve and AUPRC
pr <- pr.curve(scores.class0 = scores[true_labels == 1],
               scores.class1 = scores[true_labels == 0],
               curve = TRUE)

print(pr$auc.integral)
plot(pr)



###############################################################################
####################### 5. Logistic Regression (SMOTE) ########################
###############################################################################

library(smotefamily)
library(caret)
library(dplyr)
library(pROC)

# Step 1: Train-Test Split BEFORE SMOTE
set.seed(44940076)
train_indices <- sample(seq_len(nrow(credit_sample)), 
                        size = 0.7 * nrow(credit_sample))
train_data <- credit_sample[train_indices, ]
test_data  <- credit_sample[-train_indices, ]

# Step 2: Convert outcome to numeric
train_data$status <- as.numeric(as.character(train_data$status))
test_data$status  <- as.numeric(as.character(test_data$status))

# Step 3: Apply SMOTE (only to training)
smote_input <- train_data %>% dplyr::select(-time)

smote_result <- SMOTE(
  X       = smote_input[, -ncol(smote_input)],
  target  = smote_input$status,
  K       = 5,
  dup_size = 3
)

train_data_smote <- smote_result$data

# Ensure outcome is numeric 0/1
train_data_smote$status <- as.numeric(as.character(train_data_smote$class))
train_data_smote$class <- NULL

# Step 4: Fit Logistic Regression
logit_smote <- glm(status ~ ., data = train_data_smote, family = binomial)

# Step 5: Predict on untouched test data
test_probs <- predict(logit_smote, newdata = test_data, type = "response")
test_pred  <- ifelse(test_probs > 0.5, 1, 0) # Change the scale from 0.05 to a threshold of choice.

# Step 6: Confusion Matrix
conf_matrix_smote <- confusionMatrix(
  factor(test_pred),
  factor(test_data$status),
  positive = "1"
)
print(conf_matrix_smote)


###############################################################################
# ROC + AUC
###############################################################################
# Step 7: ROC + AUC
roc_logit_smote <- roc(response = test_data$status, predictor = test_probs)
plot(roc_logit_smote, 
     col = "red", 
     lwd = 2, 
     main = "ROC Curve — Logistic Regression (With SMOTE)")
abline(a = 0, b = 1, lty = 2)

auc_logit_smote <- auc(roc_logit_smote)
print(paste("AUC (With SMOTE):", round(auc_logit_smote, 4)))

###############################################################################
# AUPRC
###############################################################################
# Step 8: Calculate predicted probabilities from logit model.
scores <- test_probs  

# Step 9: Compute PR curve and AUPRC
pr <- pr.curve(scores.class0 = scores[true_labels == 1],
               scores.class1 = scores[true_labels == 0],
               curve = TRUE)

print(pr$auc.integral)
plot(pr)


###############################################################################
##################### 6. Random Survival Forest (no SMOTE) ####################
###############################################################################
# Step 1: Train-test split
set.seed(44940076)
train_indices <- sample(seq_len(nrow(credit_sample)), 
                        size = 0.7 * nrow(credit_sample))
train_data <- credit_sample[train_indices, ]
test_data  <- credit_sample[-train_indices, ]

# Step 2: Fit RSF on training data
library(randomForestSRC)
gc()

rsf_model_train <- rfsrc(Surv(time, status) ~ ., data = train_data,
                         ntree = 100,
                         nodesize = 15,
                         splitrule = "logrank",
                         nthread = 1,
                         fast = TRUE)
# Step 3: Predict
rsf_pred <- predict(rsf_model_train, newdata = test_data)
mortality_test <- rsf_pred$predicted[, "event.1"]


# Step 4: Check lengths
stopifnot(length(mortality_test) == nrow(test_data))

# Step 5: Classify
threshold <- median(mortality_test)
pred_class <- ifelse(mortality_test > threshold, 1, 0)

# Step 6: Evaluate
conf_matrix_rsf_test <- confusionMatrix(factor(pred_class), 
                                        factor(test_data$status), 
                                        positive = "1")
print(conf_matrix_rsf_test)




###############################################################################
# ROC + AUC
###############################################################################
# Step 6: Load required package
#install.packages("pROC")
library(pROC)

# Step 7: Use predicted mortality as risk scores
mortality_test <- rsf_pred$predicted[, "event.1"]
true_class <- test_data$status

# Step 8: Generate ROC curve
roc_rsf <- roc(response = true_class, predictor = mortality_test)

# Step 9: Plot ROC curve
plot(roc_rsf, 
     col = "cornflowerblue", 
     lwd = 2, 
     main = "ROC Curve — RSF without SMOTE")
abline(a = 0, b = 1, lty = 2, col = "gray")

# Step 10: AUC
auc_rsf <- auc(roc_rsf)
print(paste("AUC:", round(auc_rsf, 4)))

###############################################################################
# AUPRC
###############################################################################
# Step 11: True test labels (must be 0-1 numeric)
true_class <- test_data$status

# Step 12: Compute Precision-Recall Curve and AUPRC
pr <- pr.curve(
  scores.class0 = mortality_test[true_class == 1],
  scores.class1 = mortality_test[true_class == 0],
  curve = TRUE
)

print(paste("AUPRC:", round(pr$auc.integral, 4)))
plot(pr)



###############################################################################
###################### 7. Random Survival Forest (SMOTE) ######################
###############################################################################

# Step 1: Train-test split BEFORE SMOTE
set.seed(44940076)
train_idx <- sample(seq_len(nrow(credit_sample)), 
                    size = 0.7 * nrow(credit_sample))
train_data <- credit_sample[train_idx, ]
test_data  <- credit_sample[-train_idx, ]

train_data$status <- as.numeric(as.character(train_data$status))
test_data$status  <- as.numeric(as.character(test_data$status))

# Step 2: Apply SMOTE ONLY to predictors + status
library(smotefamily)
library(dplyr)

# Convert to numeric
train_data$status <- as.numeric(as.character(train_data$status))

# SMOTE wants predictors only — KEEP real time aside
train_X <- train_data %>% dplyr::select(-status, -time)
train_y <- train_data$status

# Apply SMOTE
smote_out <- SMOTE(
  X = train_X,
  target = train_y,
  K = 5,
  dup_size = 3
)

# SMOTE returns synthetic X + new 'class' -> rename and merge real time
train_data_smote <- smote_out$data
train_data_smote$status <- as.numeric(train_data_smote$class)
train_data_smote$class <- NULL

# Reattach the REAL times (important for survival)
train_data_smote$time <- NA_real_

idx0 <- which(train_data_smote$status == 0)
idx1 <- which(train_data_smote$status == 1)

train_data_smote$time[idx0] <- sample(
  train_data$time[train_data$status == 0],
  length(idx0),
  replace = TRUE
)

train_data_smote$time[idx1] <- sample(
  train_data$time[train_data$status == 1],
  length(idx1),
  replace = TRUE
)

# Step 3: Fit RSF
library(randomForestSRC)

rsf_model_smote <- rfsrc(
  Surv(time, status) ~ .,
  data = train_data_smote,
  ntree = 100,
  nodesize = 15,
  splitrule = "logrank",
  nthread = 1,
  fast = TRUE
)

# Step 4: Predict on untouched test data
rsf_pred <- predict(rsf_model_smote, newdata = test_data)

mortality_test <- 1 - rsf_pred$survival[, ncol(rsf_pred$survival)]
mortality_test <- as.numeric(mortality_test)

# Step 5: Classify based on mortality threshold
threshold <- median(mortality_test)
pred_class <- ifelse(mortality_test > threshold, 1, 0)

# Step 6: Confusion matrix
library(caret)
conf_matrix_rsf_smote <- confusionMatrix(
  factor(pred_class),
  factor(test_data$status),
  positive = "1"
)

print(conf_matrix_rsf_smote)



###############################################################################
# ROC + AUC
###############################################################################
# Step 7: Generate ROC curve.
library(pROC)

roc_rsf <- roc(
  response = test_data$status,
  predictor = mortality_test
)

plot(roc_rsf, col = "magenta", lwd = 2,
     main = "ROC Curve — RSF with SMOTE")
abline(a = 0, b = 1, lty = 2, col = "gray")

auc_rsf <- auc(roc_rsf)
print(paste("AUC:", round(auc_rsf, 4)))

###############################################################################
# AUPRC
###############################################################################
# Step 8: True test labels (must be 0-1 numeric)
true_class <- test_data$status

# Step 9: Compute Precision-Recall Curve and AUPRC
pr <- pr.curve(
  scores.class0 = mortality_test[true_class == 1],
  scores.class1 = mortality_test[true_class == 0],
  curve = TRUE
)

print(paste("AUPRC:", round(pr$auc.integral, 4)))
plot(pr)



###############################################################################
# Sensitivity & AUPRC
###############################################################################
# Step 10: Dot plot
library(ggplot2)

df <- data.frame(
  Model = c("RSF (SMOTE)", "LogReg (SMOTE)", "LogReg (No)", "RSF (No)"),
  Sensitivity = c(0.975, 0.785, 0.722, 0.158),
  AUPRC = c(0.860, 0.853, 0.849, 0.007)
)

# Order so best is on top
df$Model <- factor(df$Model, levels = df$Model)

ggplot(df, aes(x = Sensitivity, y = Model)) +
  geom_point(size = 4) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
  labs(x = "Sensitivity (Recall)", y = NULL) +
  theme_minimal(base_size = 14)

# Step 11: Bar plot
library(ggplot2)

df <- data.frame(
  Model = c("RSF (SMOTE)", "LogReg (SMOTE)", "LogReg (No)", "RSF (No)"),
  Sensitivity = c(0.975, 0.785, 0.722, 0.158)
)

df$Model <- factor(df$Model, levels = rev(df$Model))

# Flag best model
df$Highlight <- ifelse(df$Model == "RSF (SMOTE)", "RSF (SMOTE)", "Other")

ggplot(df, aes(x = Model, y = Sensitivity, fill = Highlight)) +
  geom_col(width = 0.7) +
  geom_text(
    aes(label = sprintf("%.3f", Sensitivity)),
    hjust = -0.1,
    size = 4
  ) +
  coord_flip() +
  scale_y_continuous(
    limits = c(0, 1.05),
    breaks = seq(0, 1, 0.2)
  ) +
  scale_fill_manual(
    values = c("RSF (SMOTE)" = "#C0392B", "Other" = "grey70")
  ) +
  labs(
    x = NULL,
    y = "Sensitivity (Recall)",
    title = 'Sensitivity under Extreme Class Imbalance'
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    axis.text.y = element_text(color = "black"),
    axis.text.x = element_text(color = "black")
  )


###############################################################################
######################## 8. SHAP for RSF with SMOTE ###########################
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
######################## 9. LIME for RSF with SMOTE ###########################
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
