# Student Cost of Living Intelligence System

Survey of 50 students on the Bidholi and Kandoli campuses (Dehradun): monthly allowance, side income, four spending categories, housing, locality and self-rated financial stress (1-10).

**Run everything:** `pip install -r requirements.txt && python run_all.py` (about 1 minute). R part: `Rscript r/analysis.R`.

---

## 1. Project structure

| Path | What it is |
|---|---|
| `part1_pipeline_cleaning.py` | Extract → clean → wrangle pipeline, quality audit, feature construction, SQLite DB, data catalog, EDA plots |
| `part2_statistics_optimization.py` | Probability, distributions, CLT, estimation, hypothesis tests, ANOVA, gradient descent, PCA, feature selection, constrained budget optimiser |
| `part3_machine_learning.py` | Regression and classification (6 models each), K-Means / Hierarchical / DBSCAN / GMM, t-SNE, Apriori (from scratch), anomaly detection |
| `r/analysis.R` | Unit VI: the same analysis in R |
| `data/` | `v0_raw.csv` → `v1_clean.csv` → `v2_features.csv` → `v3_model_outputs.csv`, plus `student_cost_clean.xlsx` |
| `db/student_cost.db` | Relational database (3 tables, index) |
| `results/` | 20+ result tables, `data_catalog.json`, `cleaning_log.csv`, `sql_query_results.md` |
| `figures/` | 27 figures |

---

## 2. Data cleaning (what was actually wrong with the file)

| # | Problem found | Action | Rows |
|---|---|---|---|
| 1 | Inconsistent locality labels (`bidholi`, `nanda ki chowki`, `Prem nagar␣`) | Trim + title-case: 12 → 9 distinct values | 3 |
| 2 | `Shared Flat / PG` duplicates `PG (Shared)` | Merged | 1 |
| 3 | Gender missing (1) / "Prefer not to say" (1) | Merged into `Undisclosed`, rows kept | 2 |
| 4 | Missing Allowance, Commute, Stress (1 each) | Median of same housing group + `*_Imputed` flag column | 3 |
| 5 | Reported total ≠ sum of the four items by >25% | Flagged; **itemised sum used as the target** | 5 |
| 6 | Year 4 student aged 19 (implausible) | Flagged, not deleted | 1 |
| 7 | Tukey-fence outliers in money columns | Flagged and **kept** (real high earners, not errors) | 11 |
| 8 | Hostel and home students have Rent = 0 | Flagged, see limitation 1 | 12 |
| 9 | Duplicates, negative values, out-of-range stress | None found | 0 |

Quality audit: completeness 99.5% → 100%, label uniformity 94% → 100%. Each step is in `results/cleaning_log.csv`.

---

## 3. Syllabus coverage

### Unit I: Introduction to Data Science
| Topic | Where |
|---|---|
| Real-world application, challenges | Cost-of-living problem; small-n, self-reported estimates, hostel-fee gap (section 6) |
| Descriptive / diagnostic / predictive / prescriptive analytics | Descriptive: EDA, summaries. Diagnostic: ANOVA, correlations, why PG students are stressed. Predictive: cost regression, stress classifier. Prescriptive: constrained budget optimiser |
| CRISP-DM | Business understanding → data understanding (audit, EDA) → preparation (cleaning, features) → modelling (Parts 2 and 3) → evaluation (cross-validation, Holm correction) → deployment (reusable scripts, DB, budget planner) |
| DataOps | One-command `run_all.py`, versioned data files with hashes, logged pipeline stages |

### Unit II: Probability and Statistics
| Topic | Implementation |
|---|---|
| Conditional probability, independence | P(stress \| housing), P(deficit \| housing), Bayes' rule, χ² and Fisher exact tests (`10_conditional_probability.png`) |
| Random variables, distributions, PDF/CDF | Normal, lognormal, gamma, Weibull fitted and compared by AIC and KS test; Q-Q plot; Binomial for side-income prevalence (`11_distribution_fitting.png`) |
| Expectation, moments | Mean, variance, skewness, kurtosis |
| Random vectors, joint/marginal/conditional, covariance, correlation | Mean vector and covariance of the 4 cost categories, E[Food \| Rent], Mahalanobis multivariate-normality check, 28 correlation tests (Pearson and Spearman) |
| Central limit theorem | Simulation at n = 2, 10, 30 against theory (`13_clt.png`) |
| Sampling distributions, point/interval estimation | χ² CI for variance, t / z / bootstrap CIs for the mean, Wilson CIs for proportions |
| Hypothesis testing, one- and two-tailed | Welch t, F-test, one-tailed t, one-sample t, z-test for a proportion, paired t |
| ANOVA, one-way and two-way | One-way (housing, stream, year), Tukey HSD, Kruskal-Wallis, two-way Campus × Housing with interaction |
| Multiple comparisons | 43 tests run, Holm-corrected (`results/all_tests_holm.csv`) |

### Unit III: Data, Sources, Visualisation
| Topic | Implementation |
|---|---|
| Data quality and issues | Four-dimension audit before/after (completeness, uniqueness, validity, consistency) |
| Data pipeline stages | Extract → clean → wrangle → explore, each timed and logged (`data_catalog.json`) |
| Relational databases | SQLite with `student`, `expense`, `locality_dim`; 6 SQL queries including JOIN, HAVING, subquery, window `RANK()` |
| Data versioning, cataloguing | v0 → v3 files with SHA-256 prefixes; data dictionary and cleaning log in the catalog |
| Data transformation, feature management | Section Unit IV below |
| Visualisation for EDA | 27 figures: distributions, boxplots, heatmap, stacked spend, locality ranking, scatter, dendrogram, t-SNE… |
| Warehouse, lake, streaming/batch ingestion | Conceptual only: a 50-row survey does not need them. The pipeline is written as batch ingestion and could be pointed at a stream |

### Unit IV: Feature Engineering and Optimisation
| Topic | Implementation |
|---|---|
| Feature construction | 20+ new features: Total_Income, Net_Balance, Savings_Rate, Spend_to_Income, category shares, Essential_Ratio, Cost_Index, Wellbeing_Score, Deficit, High_Stress |
| Feature subset selection | Mutual information + correlation filter, forward selection by adjusted R² |
| Feature reduction | PCA (4 components keep 80% of the variance), loadings, scree plot |
| Feature learning | PCA and t-SNE embeddings reused as features |
| Text / image / audio features | Not applicable (no such data in the survey) |
| Gradient descent | Batch GD, momentum GD and mini-batch SGD written from scratch; match the closed-form least-squares answer to within 3·10⁻⁵ (`15_gradient_descent.png`) |
| Least squares | Normal equation, `lstsq`, sklearn: all agree |
| Constrained optimisation | SLSQP: best budget split per housing type and budget under floors and a budget cap (`17_optimal_budget.png`) |
| Non-gradient technique | Nelder-Mead maximum-likelihood fit of the lognormal, matching the closed form |

### Unit V: Supervised and Unsupervised Learning
| Topic | Implementation |
|---|---|
| Supervised workflow | Pipelines (scaling + one-hot), repeated k-fold CV (5×10), baseline model |
| Regression | Linear, Ridge, Decision Tree, Random Forest, SVR, k-NN |
| Classification | Logistic, Decision Tree, Random Forest, SVM, k-NN; ROC, confusion matrix, k tuning |
| Clustering | K-Means (elbow, silhouette, Davies-Bouldin, Calinski-Harabasz), Ward hierarchical (dendrogram, cophenetic), DBSCAN (k-distance eps), Gaussian Mixture (BIC) |
| Dimensionality reduction | PCA, t-SNE |
| Association rules | **Apriori implemented from scratch** (support, confidence, lift, leverage, conviction). FP-Growth not implemented: it finds the same rules, only faster, which is irrelevant at n = 50 |
| Anomaly detection | Isolation Forest, LOF, Mahalanobis, robust z-score → consensus vote |
| Model evaluation | R², MAE, RMSE, accuracy, precision, recall, F1, ROC-AUC, silhouette, inertia, ARI between clusterers |

### Unit VI: R
`r/analysis.R` covers reading files, ordered and unordered factors, vectors, matrices, a 3-D array, lists and data frames, `split`/`lapply`/`aggregate`/`merge`/`tapply`, `d/p/q/r` distribution functions, `t.test`, `var.test`, `chisq.test`, `aov` + `TukeyHSD`, `lm`, `glm`, `prcomp`, `kmeans`, `hclust`. **R is not installed in my build environment, so this script was not executed**; run it once and fix anything the first run flags.

---

## 4. Main findings

1. **Housing type is the main cost driver.** It explains 67% of the variance in monthly cost (one-way ANOVA F = 31.4, η² = 0.67). Flat and PG students spend ₹14,000-19,000 more than hostel and home students (Tukey p < 0.001). PG vs Flat is not significantly different (p = 0.38).
2. **The Bidholi vs Kandoli gap is mostly a housing effect.** Bidholi averages ₹22,446 vs ₹14,568 (Welch p = 0.001, d ≈ 1.0), but in the two-way ANOVA the Campus effect disappears once housing is included (p = 0.44, interaction p = 0.96). Bidholi has more students in paid accommodation.
3. **Average cost is ₹18,980/month** (95% CI ₹16,490-21,470), significantly above a ₹15,000 benchmark (p = 0.002).
4. **38% of students spend more than they receive** (95% CI 26-52%). Among deficit students 58% report high stress vs 10% of the others (Fisher OR = 12.8, p = 0.0007; survives Holm correction).
5. **PG students carry the stress.** 12 of the 14 high-stress students live in PG housing. High stress: PG 48%, Flat 20%, Hostel 17%, Home 0% (χ² p = 0.013; some expected counts are below 5, so read with care).
6. **Four student personas** (K-Means, k = 4): *Hostel/Home savers* (n = 18, avg cost ₹9.3k, savings rate +30%), *Balanced off-campus* (n = 20, ₹23k, about break-even), *Squeezed PG students* (n = 8, all but one in PG, all Bidholi, no side income, stress 8.9, spending about 48% above income), *High-spend earners* (n = 4, ₹33k, side income ₹11.7k).
7. **Rules:** `Deficit & Low income → High stress` (confidence 0.86, lift 3.1).
8. **Anomalies (≥3 of 4 detectors agree):** IDs 4, 18, 14, 22, 35. ID 18 (day scholar, ₹25k side income, ₹5k allowance, but also the Year-4/age-19 flag) deserves a data-entry check.
9. **Prediction:** cost can be predicted from context with cross-validated R² = 0.73 (Ridge, MAE ≈ ₹3,400); allowance alone dominates. High stress is predicted with ROC-AUC 0.81 (Random Forest) / 0.78 (Logistic), against 0.50 for the baseline.
10. **Budget planner:** for a ₹20,000 PG budget the optimiser suggests roughly Rent ₹9,650 / Food ₹5,400 / Commute ₹1,350 / Academics ₹3,650 (`results/optimal_budget_plan.csv`).

---

## 5. Honest limitations

1. **Hostel fees are not in the data.** All 12 hostel students report Rent = 0, so hostel "cost" is understated and the hostel-vs-PG gap is overstated as a true cost difference. The finding says "reported out-of-pocket cost", not "total cost of living". Add the fee column if you can get it.
2. **n = 50.** Several groups have 1-4 students (Home n = 3, Design n = 4). The cross-validated scores have wide spread (R² SD ≈ 0.16, AUC SD ≈ 0.14), and cluster separation is weak (silhouette ≈ 0.24). Treat clusters as descriptive personas, not hard categories.
3. **Allowance is endogenous.** It largely determines spending, so "Allowance predicts cost" is partly circular. Models are therefore reported with context features and a baseline.
4. **Stress classifier recall is modest** (about 0.55-0.61 depending on model): it finds roughly 3 in 5 high-stress students. Accuracy (0.84 for Logistic) looks better than it is because 72% of students are not high-stress.
5. **Multiple testing:** 43 tests were run; 10 remain significant after Holm correction (listed in `results/all_tests_holm.csv`). The rest should not be quoted as findings.
6. **Self-reported estimates** ("Est_INR"): measurement noise is built in (the 5 mismatched totals).

---

## 6. Suggested next steps

- Add a hostel-fee column and re-run (everything reruns from `run_all.py`).
- Collect more responses (aim for 150+); then the cluster and classifier results become much more reliable.
- Turn `results/optimal_budget_plan.csv` and the peer-matching k-NN (leave-one-out R² 0.66) into a small web or Streamlit budget calculator.
