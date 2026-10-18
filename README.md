# When Do Random Survival Forests Enhance Fraud Detection? A Conditions-Based Comparison

Code and analysis notebooks for my Master of Research (MRes) thesis at Macquarie University (2026).

**Author:** Arjun Sekhar

**Supervisors:** Prof. Samuel Muller and A/Prof. Maurizio Manuguerra

---

## Overview

Random Survival Forests (RSF) model the *time* to an event, not just whether it happens. That makes them a tempting candidate for fraud detection. This project asks **under what conditions an RSF actually outperforms Logistic Regression (LR)** for detecting card fraud under extreme class imbalance. It does not simply assume the survival framing helps.

The comparison is organised around a **conditions matrix**. Drawn from the comparative survival literature, it maps the predictor structure, censoring rate and event count under which RSF is expected to gain or lose against parametric alternatives. Cox Proportional Hazards and a Random Forest (RF) classification control are added to separate the effect of the survival framing from that of the tree-ensemble structure.

### Key findings

- **ULB credit card data:** LR and RSF performed almost identically (AUPRC within 0.01) across four configurations, with and without SMOTE. This matches what the conditions matrix predicts for PCA-transformed predictors, about 99.8% censoring and a modest event count.
- **Controlled simulation:** the relationship reverses as predicted. LR wins under ULB-like conditions, and RSF wins once covariate interactions and moderate censoring are introduced.
- **Sparkov account-level data:** RSF outperformed LR and Cox, but the simulator assigns fraud timing independently of customer attributes. The comparison therefore tests detection of a scripted fraud signature, not a survival advantage.
- **Across every evaluation,** the RF classification control matched or beat RSF. Tree-ensemble flexibility, not survival framing, was the larger driver of performance.

---

## Repository structure

| Folder | Thesis section | Contents |
|---|---|---|
| [`1-ULB_MultiMethod`](1-ULB_MultiMethod) | 3.1–3.6, 4.2–4.5.2, 4.7, 4.8 | Primary ULB analysis: LR, RSF, RF and Cox; SMOTE configurations; full-dataset robustness check; label-reassignment simulation; concordance diagnostic |
| [`2-ULB Time Sensitivity`](2-ULB%20Time%20Sensitivity) | 4.8.2 | Paired refits of every model with and without the raw `Time` covariate (final version: **v4**) |
| [`3-ULB Explainability`](3-ULB%20Explainability) | 4.6 | Variable importance, SHAP, LIME and cross-model importance rankings |
| [`4-Simulations/DGP Scenario Reproduction`](4-Simulations/DGP%20Scenario%20Reproduction) | 4.5.1, 5.4 | Controlled data-generating-process simulation (Scenario A: ULB-like; Scenario B: favourable to RSF) |
| [`4-Simulations/NHPP Simulation`](4-Simulations/NHPP%20Simulation) | (exploratory) | Non-homogeneous Poisson process simulations varying the strength of a daily fraud cycle |
| [`5-Sparkov (Account Level)`](5-Sparkov%20(Account%20Level)) | 3.7, 4.9 | Account-level survival analysis, fraud-structure checks, window sensitivity, explainability and fraud-amount analysis |
| [`6-Aggregate Temporal (Poisson Diurnal)`](6-Aggregate%20Temporal%20(Poisson%20Diurnal)) | 3.8, 4.10 | Poisson and Negative Binomial models of hourly and weekly fraud-rate cycles |
| [`Archive`](Archive) | – | Earlier versions, kept for transparency; not used for the final results |

Each folder contains the R Markdown notebook (`.Rmd`), its knitted PDF, and the cached results (`.rds`) behind the reported figures.

---

## Data

The datasets are **not included** in this repository because of their size. Both are publicly available on Kaggle:

| Dataset | Source | File(s) needed |
|---|---|---|
| ULB Credit Card Fraud (Dal Pozzolo et al., 2015) | [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `creditcard.csv` |
| Sparkov Credit Card Transactions (Shenoy, 2020) | [kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) | `fraudTrain.csv`, `fraudTest.csv` |

Download the files and place them in the **same folder as the notebook** that uses them. Folders 1, 2, 3 and 6 use `creditcard.csv`, and folder 5 uses the two Sparkov files.

The Sparkov data was produced by the open-source [Sparkov Data Generation](https://github.com/namebrandon/Sparkov_Data_Generation) simulator (Harris, 2020).

---

## Reproducing the results

**Software:** R with the following packages:

```r
install.packages(c(
  "randomForestSRC", "survival", "glmnet", "randomForest", "caret",
  "smotefamily", "pROC", "PRROC", "iml", "lime", "MASS", "lmtest",
  "sandwich", "dplyr", "tidyr", "lubridate", "pander", "rmarkdown"
))
```

**Notes:**

- **Seed:** all analyses use `set.seed(44940076)`. Repeated splits use `44940076 + i` where stated.
- **Cached results:** the notebooks load saved `.rds` results if they exist and refit the models only if they don't. Delete an `.rds` file to force a full re-run of that analysis.
- **Run time:** the primary configurations knit in minutes. The full-dataset RSF (SMOTE) robustness check (284,807 transactions) can take several hours.
- **Evaluation:** unless stated otherwise, results are means and standard deviations across 30 repeated stratified 70/30 train-test splits. AUPRC is the primary metric.

---

## Archive

The `Archive` folder holds superseded material:

- `Fraud Detection - Methodology Notebook v2.Rmd`: an earlier name for the primary ULB notebook (identical to the version in folder 1).
- `Fraud Detection - Methodology Notebook v3.Rmd`: an exploratory variant that removed the raw `Time` covariate. It was **not** used in the thesis. The question it raised is answered by the Time Sensitivity analysis in folder 2.
- `RSF_with_CCFD_FINAL.R`: the original script version of the analysis, before it moved to R Markdown.
- `RSF_Reproduction_Ishwaran2007.R`: a reproduction of the original RSF method (Ishwaran & Kogalur), used to learn the method.

---

## Citation

If you use this code, please cite:

> Sekhar, A. (2026). *When do random survival forests enhance fraud detection? A conditions-based comparison* [Master of Research thesis, Macquarie University].

## License

[MIT](LICENSE)
