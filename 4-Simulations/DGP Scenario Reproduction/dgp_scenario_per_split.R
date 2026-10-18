## =====================================================================
## DGP scenario reproduction: per-replicate results
##
## Why this script exists: run_scenario() in DGP_Scenario_Reproduction.Rmd
## returns only the means and SDs, so dgp_scenario_results.rds holds two
## summary rows and no per-replicate values. This script repeats the same
## simulation, with the same seed, functions and order, but keeps every
## replicate so you can count how often each method came out ahead.
##
## Each of the 30 iterations generates a NEW dataset from the DGP and splits
## it once (stratified 70/30). So these are 30 independent simulated datasets,
## not 30 repeated splits of one dataset.
##
## Runtime: about 10-15 minutes. Run from the folder that holds
## dgp_scenario_results.rds (for the reproducibility check at the end).
## =====================================================================

## Step 1: Libraries
library(randomForestSRC)
library(survival)
library(PRROC)
library(pander)

## Step 2: Same seed and function definitions as the notebook
set.seed(44940076)

generate_survival_data <- function(n, p = 10, betas,
                                   interactions = NULL,
                                   censoring_rate = 0.99) {
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  colnames(X) <- paste0("X", 1:p)
  lp <- X %*% betas
  if (!is.null(interactions)) {
    for (term in interactions) {
      lp <- lp + term$gamma * X[, term$i] * X[, term$j]
    }
  }
  lambda <- exp(lp)
  time_event <- rexp(n, rate = lambda)
  lambda_c <- quantile(time_event, probs = 1 - censoring_rate)
  time_censor <- rexp(n, rate = 1 / as.numeric(lambda_c))
  time_obs <- pmin(time_event, time_censor)
  status <- as.integer(time_event <= time_censor)
  data.frame(X, time = time_obs, status = status)
}

## Step 3: Scenario runner. Identical to the notebook except that it also
## returns the per-replicate values. Nothing that draws random numbers has
## changed, so the random stream is the same as in the original run.
run_scenario <- function(scenario_label, n, betas,
                         interactions = NULL,
                         censoring_rate,
                         n_splits = 30) {
  auprc_lr  <- numeric(n_splits)
  auprc_rsf <- numeric(n_splits)
  cerr_lr   <- numeric(n_splits)
  cerr_rsf  <- numeric(n_splits)
  n_events  <- numeric(n_splits)

  for (i in seq_len(n_splits)) {
    message(" [", scenario_label, "] Replicate ", i, " of ", n_splits, " - ", Sys.time())

    dat <- generate_survival_data(n, betas = betas,
                                  interactions = interactions,
                                  censoring_rate = censoring_rate)
    n_events[i] <- sum(dat$status)

    event_idx    <- which(dat$status == 1)
    nonevent_idx <- which(dat$status == 0)
    train_idx <- c(
      sample(event_idx,    size = floor(0.7 * length(event_idx))),
      sample(nonevent_idx, size = floor(0.7 * length(nonevent_idx)))
    )
    train <- dat[train_idx, ]
    test  <- dat[-train_idx, ]

    lr_mod  <- glm(status ~ . - time, data = train, family = binomial)
    lr_prob <- predict(lr_mod, newdata = test, type = "response")

    rsf_mod <- rfsrc(Surv(time, status) ~ ., data = train,
                     ntree = 100, nodesize = 15,
                     splitrule = "logrank", fast = TRUE, nthread = 4)
    rsf_pred <- predict(rsf_mod, newdata = test[, rsf_mod$xvar.names, drop = FALSE])
    rsf_risk <- 1 - rsf_pred$survival[, ncol(rsf_pred$survival)]
    rsf_risk <- as.numeric(rsf_risk)

    true_labels <- test$status

    lr_keep  <- is.finite(lr_prob)  & is.finite(true_labels)
    rsf_keep <- is.finite(rsf_risk) & is.finite(true_labels)

    auprc_lr[i] <- pr.curve(
      scores.class0 = lr_prob[lr_keep][true_labels[lr_keep] == 1],
      scores.class1 = lr_prob[lr_keep][true_labels[lr_keep] == 0],
      curve = FALSE)$auc.integral
    auprc_rsf[i] <- pr.curve(
      scores.class0 = rsf_risk[rsf_keep][true_labels[rsf_keep] == 1],
      scores.class1 = rsf_risk[rsf_keep][true_labels[rsf_keep] == 0],
      curve = FALSE)$auc.integral

    ## "CErr" is 1 - concordance() on risk scores, which recovers the C-index
    ## (values above 0.5 mean good discrimination), as noted in the notebook.
    cerr_lr[i]  <- 1 - as.numeric(concordance(Surv(test$time, test$status) ~ lr_prob)$concordance)
    cerr_rsf[i] <- 1 - as.numeric(concordance(Surv(test$time, test$status) ~ rsf_risk)$concordance)

    rm(rsf_mod, rsf_pred, dat, train, test)
    gc()
  }

  list(
    summary = data.frame(
      Scenario = scenario_label,
      AUPRC_LR = mean(auprc_lr),   AUPRC_LR_SD = sd(auprc_lr),
      AUPRC_RSF = mean(auprc_rsf), AUPRC_RSF_SD = sd(auprc_rsf),
      CErr_LR = mean(cerr_lr),     CErr_LR_SD = sd(cerr_lr),
      CErr_RSF = mean(cerr_rsf),   CErr_RSF_SD = sd(cerr_rsf),
      Events_mean = round(mean(n_events)), Events_SD = round(sd(n_events))
    ),
    replicates = data.frame(
      Scenario = scenario_label, replicate = seq_len(n_splits),
      auprc_lr = auprc_lr, auprc_rsf = auprc_rsf,
      cindex_lr = cerr_lr, cindex_rsf = cerr_rsf, n_events = n_events
    )
  )
}

betas_A <- c(0.5, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05)

## Step 4: Run both scenarios in the same order as the notebook (A, then B)
res_A <- run_scenario(
  scenario_label = "A: Linear additive, high censoring",
  n = 10000, betas = betas_A, interactions = NULL, censoring_rate = 0.99
)
res_B <- run_scenario(
  scenario_label = "B: Interactions, moderate censoring",
  n = 5000, betas = betas_A,
  interactions = list(list(i = 1, j = 3, gamma = 0.5),
                      list(i = 2, j = 4, gamma = 0.5)),
  censoring_rate = 0.50
)

dgp_summary    <- rbind(res_A$summary, res_B$summary)
dgp_replicates <- rbind(res_A$replicates, res_B$replicates)
saveRDS(dgp_replicates, "dgp_scenario_per_replicate.rds")

## Step 5: Reproducibility check against the saved summary. If the rerun
## reproduces the notebook, these differences are 0 (or tiny). If they are not,
## use the rerun's numbers in the thesis table and text so everything agrees.
if (file.exists("dgp_scenario_results.rds")) {
  old <- readRDS("dgp_scenario_results.rds")
  num_cols <- c("AUPRC_LR", "AUPRC_LR_SD", "AUPRC_RSF", "AUPRC_RSF_SD",
                "CErr_LR", "CErr_LR_SD", "CErr_RSF", "CErr_RSF_SD", "Events_mean")
  print(round(as.matrix(dgp_summary[, num_cols]) - as.matrix(old[, num_cols]), 6))
}
pander(dgp_summary, digits = 4, caption = "Rerun summary (compare with the notebook table)")

## Step 6: Counts and paired Wilcoxon tests, per scenario
count_table <- do.call(rbind, lapply(split(dgp_replicates, dgp_replicates$Scenario), function(d) {
  w_auprc <- wilcox.test(d$auprc_rsf,  d$auprc_lr,  paired = TRUE)
  w_c     <- wilcox.test(d$cindex_rsf, d$cindex_lr, paired = TRUE)
  data.frame(
    Scenario        = d$Scenario[1],
    n_replicates    = nrow(d),
    auprc_rsf_ahead = sum(d$auprc_rsf > d$auprc_lr),
    auprc_lr_ahead  = sum(d$auprc_lr  > d$auprc_rsf),
    auprc_p         = w_auprc$p.value,
    cindex_rsf_ahead = sum(d$cindex_rsf > d$cindex_lr),
    cindex_lr_ahead  = sum(d$cindex_lr  > d$cindex_rsf),
    cindex_p        = w_c$p.value
  )
}))
pander(count_table, digits = 3, caption = "Replicates in which each method came out ahead (30 independent simulated datasets per scenario)")
