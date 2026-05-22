library(randomForestSRC)
library(survival)

## ============================================================
## Reproduction of Ishwaran & Kogalur (2007) R News Results
## Veteran lung cancer data and PBC data
## ============================================================

## ----------------------------------------------------------
## PART 1: Veteran data — single predictor (karno only)
## Paper reports error rate: 36.28%
## ----------------------------------------------------------

data(veteran, package = "randomForestSRC")

set.seed(44940076)

## Paper parameters: ntree=1000, default nodesize=15 in rfsrc
## Note: old package used nodesize=3 but rfsrc default is 15
## We test both

## Paper parameters (as close as possible)
v_paper <- rfsrc(
  Surv(time, status) ~ karno,
  data      = veteran,
  ntree     = 1000,
  splitrule = "logrank",
  nodesize  = 3         # original paper default
)
cat("Veteran (karno only) — Paper params:\n")
cat("  Error rate:", tail(v_paper$err.rate, 1), "\n")
cat("  Paper reports: 0.3628\n\n")

## Your parameters
v_yours <- rfsrc(
  Surv(time, status) ~ karno,
  data      = veteran,
  ntree     = 100,
  splitrule = "logrank",
  nodesize  = 15,
  nthread   = 4,
  fast      = TRUE
)
cat("Veteran (karno only) — Your params:\n")
cat("  Error rate:", tail(v_yours$err.rate, 1), "\n\n")

## ----------------------------------------------------------
## PART 2: Veteran data — all 6 predictors
## 100 replications across 4 splitting rules
## Paper reports:
##   logrank:       mean 0.2982, sd 0.0027
##   conserve:      mean 0.3239, sd 0.0034
##   logrankscore:  mean 0.2951, sd 0.0027
##   logrankapprox: mean 0.3170, sd 0.0046
## ----------------------------------------------------------

set.seed(44940076)
nrep      <- 100
splitrules <- c("logrank", "conserve", "logrankscore", "logrankapprox")

## Paper parameters
cat("--- Veteran data (all predictors) — Paper params (ntree=1000, nodesize=3) ---\n")
err_paper <- matrix(NA, nrow = length(splitrules), ncol = nrep,
                    dimnames = list(splitrules, NULL))

for (j in seq_along(splitrules)) {
  message("  Splitrule: ", splitrules[j])
  for (k in 1:nrep) {
    fit <- rfsrc(
      Surv(time, status) ~ .,
      data      = veteran,
      ntree     = 1000,
      splitrule = splitrules[j],
      nodesize  = 3
    )
    err_paper[j, k] <- tail(fit$err.rate, 1)
  }
}

results_paper <- data.frame(
  Splitrule = splitrules,
  Mean      = round(rowMeans(err_paper), 4),
  SD        = round(apply(err_paper, 1, sd), 4)
)
print(results_paper)
cat("\nPaper reports:\n")
cat("  logrank: 0.2982 (0.0027)\n")
cat("  conserve: 0.3239 (0.0034)\n")
cat("  logrankscore: 0.2951 (0.0027)\n")
cat("  logrankapprox: 0.3170 (0.0046)\n\n")

## Your parameters
cat("--- Veteran data (all predictors) — Your params (ntree=100, nodesize=15, fast=TRUE) ---\n")
err_yours <- matrix(NA, nrow = length(splitrules), ncol = nrep,
                    dimnames = list(splitrules, NULL))

for (j in seq_along(splitrules)) {
  message("  Splitrule: ", splitrules[j])
  for (k in 1:nrep) {
    fit <- rfsrc(
      Surv(time, status) ~ .,
      data      = veteran,
      ntree     = 100,
      splitrule = splitrules[j],
      nodesize  = 15,
      nthread   = 4,
      fast      = TRUE
    )
    err_yours[j, k] <- tail(fit$err.rate, 1)
  }
}

results_yours <- data.frame(
  Splitrule = splitrules,
  Mean      = round(rowMeans(err_yours), 4),
  SD        = round(apply(err_yours, 1, sd), 4)
)
print(results_yours)

## ----------------------------------------------------------
## PART 3: PBC data
## Paper reports:
##   logrank:       mean 0.1703, sd 0.0014
##   conserve:      mean 0.1677, sd 0.0014
##   logrankscore:  mean 0.1719, sd 0.0015
##   logrankapprox: mean 0.1602, sd 0.0020
## ----------------------------------------------------------

data(pbc, package = "randomForestSRC")

set.seed(44940076)

## Paper parameters
cat("--- PBC data — Paper params (ntree=1000, nodesize=3) ---\n")
err_pbc_paper <- matrix(NA, nrow = length(splitrules), ncol = nrep,
                        dimnames = list(splitrules, NULL))

for (j in seq_along(splitrules)) {
  message("  Splitrule: ", splitrules[j])
  for (k in 1:nrep) {
    fit <- rfsrc(
      Surv(days, status) ~ .,
      data      = pbc,
      ntree     = 1000,
      splitrule = splitrules[j],
      nodesize  = 3
    )
    err_pbc_paper[j, k] <- tail(fit$err.rate, 1)
  }
}

results_pbc_paper <- data.frame(
  Splitrule = splitrules,
  Mean      = round(rowMeans(err_pbc_paper), 4),
  SD        = round(apply(err_pbc_paper, 1, sd), 4)
)
print(results_pbc_paper)
cat("\nPaper reports:\n")
cat("  logrank: 0.1703 (0.0014)\n")
cat("  conserve: 0.1677 (0.0014)\n")
cat("  logrankscore: 0.1719 (0.0015)\n")
cat("  logrankapprox: 0.1602 (0.0020)\n\n")

## Your parameters
cat("--- PBC data — Your params (ntree=100, nodesize=15, fast=TRUE) ---\n")
err_pbc_yours <- matrix(NA, nrow = length(splitrules), ncol = nrep,
                        dimnames = list(splitrules, NULL))

for (j in seq_along(splitrules)) {
  message("  Splitrule: ", splitrules[j])
  for (k in 1:nrep) {
    fit <- rfsrc(
      Surv(days, status) ~ .,
      data      = pbc,
      ntree     = 100,
      splitrule = splitrules[j],
      nodesize  = 15,
      nthread   = 4,
      fast      = TRUE
    )
    err_pbc_yours[j, k] <- tail(fit$err.rate, 1)
  }
}

results_pbc_yours <- data.frame(
  Splitrule = splitrules,
  Mean      = round(rowMeans(err_pbc_yours), 4),
  SD        = round(apply(err_pbc_yours, 1, sd), 4)
)
print(results_pbc_yours)

## ----------------------------------------------------------
## PART 4: Summary comparison table
## ----------------------------------------------------------

cat("\n=== SUMMARY: Paper vs Your Parameters ===\n\n")

cat("VETERAN DATA:\n")
comparison_veteran <- data.frame(
  Splitrule     = splitrules,
  Paper_Mean    = results_paper$Mean,
  Paper_SD      = results_paper$SD,
  Yours_Mean    = results_yours$Mean,
  Yours_SD      = results_yours$SD,
  Difference    = round(results_yours$Mean - results_paper$Mean, 4)
)
print(comparison_veteran)

cat("\nPBC DATA:\n")
comparison_pbc <- data.frame(
  Splitrule     = splitrules,
  Paper_Mean    = results_pbc_paper$Mean,
  Paper_SD      = results_pbc_paper$SD,
  Yours_Mean    = results_pbc_yours$Mean,
  Yours_SD      = results_pbc_yours$SD,
  Difference    = round(results_pbc_yours$Mean - results_pbc_paper$Mean, 4)
)
print(comparison_pbc)