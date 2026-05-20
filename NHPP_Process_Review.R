set.seed(123)

# -----------------------------
# PARAMETERS
# -----------------------------

n <- 50000                   # number of transactions
p <- 30                      # covariates
target_rate <- 492/50492     # fraud prevalance rate

# -----------------------------
# TRANSACTION TIMES
# -----------------------------

# simulate transaction timestamps over 30 days
days <- 2

time_hours <- runif(n, 0, 24 * days)

# hour within day
hour_of_day <- time_hours %% 24

# -----------------------------
# DAILY FRAUD CYCLE
# -----------------------------

# peak fraud around 2am (since peak hours are occurring with a 6-hour offset)
phi <- 2

# cycle amplitude
a <- 0.8

# baseline cycle
daily_cycle <- 1 + a * sin(2 * pi * (hour_of_day - phi) / 24)

# ensure positivity
daily_cycle <- pmax(daily_cycle, 0.05)

# -----------------------------
# COVARIATES
# -----------------------------

X <- matrix(rnorm(n * p), nrow = n, ncol = p)

colnames(X) <- paste0("x", 1:p)

# -----------------------------
# EFFECT SIZES
# -----------------------------

beta <- c(
  1.2, 0.9, 0.7, 0.5, 0.3,
  -0.5, -0.7, -1.0,
  rep(0, p - 8)
)

linpred <- X %*% beta

risk_multiplier <- exp(linpred)

# -----------------------------
# CALIBRATE BASE RATE
# -----------------------------

# choose baseline so expected fraud rate matches target

lambda_base <-
  target_rate / mean(daily_cycle * risk_multiplier)

# individual probabilities
prob_fraud <-
  lambda_base * daily_cycle * risk_multiplier

# safety cap
prob_fraud <- pmin(prob_fraud, 0.5)

# -----------------------------
# GENERATE FRAUDS
# -----------------------------

fraud <- rbinom(n, 1, prob_fraud)

mean(fraud)
table(fraud)

# -----------------------------
# FINAL DATASET
# -----------------------------

df <- data.frame(
  time_hours = time_hours,
  hour_of_day = hour_of_day,
  fraud = fraud,
  X
)

head(df)


# -----------------------------
# PLOT
# -----------------------------
plot(seq(0, 24, length.out = 200),
     1 + 0.8 * sin(2 * pi * (seq(0, 24, length.out = 200) - 2) / 24),
     type = "l", xlab = "Hour of day", ylab = "Fraud rate multiplier",
     main = "Daily fraud cycle (a = 0.8, peak at 2am)")
abline(h = 1, lty = 2)


hist(df$hour_of_day[df$fraud == 1], breaks = 24,
     main = "Hour of day for fraud cases",
     xlab = "Hour of day")


# Theoretical cycle
hours <- seq(0, 24, length.out = 200)
cycle <- 1 + 0.8 * sin(2 * pi * (hours - phi) / 24)

par(mfrow = c(1, 2))

# Left: theoretical
plot(hours, cycle, type = "l", lwd = 2,
     main = "Theoretical daily fraud cycle",
     xlab = "Hour of day", ylab = "Rate multiplier")
abline(h = 1, lty = 2, col = "red")

# Right: empirical
hist(df$hour_of_day[df$fraud == 1], breaks = 24,
     main = "Observed fraud cases by hour",
     xlab = "Hour of day", col = "cornflowerblue")

# ==============================================================================
# Research Methodology Evaluation & Modelling Session
# Location: University of Western Australia (UWA), Reid Library Ground Floor
# Date: Wednesday, 20 May 2026
# Network Anchor: eduroam authentication via Macquarie University
# ==============================================================================
