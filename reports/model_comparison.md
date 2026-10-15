# Model comparison

Dataset: `data/processed/dataset.csv`, 2788 commits.
Time-ordered split: **train** 2230 commits (2011-02-13 → 2016-06-27, bug rate 23.6%),
**test** 558 commits (2016-06-28 → 2025-08-18, bug rate 15.0%).
Cross-validation: TimeSeriesSplit(n_splits=5) on train; tuning: RandomizedSearchCV(n_iter=25, scoring='roc_auc'). Thresholds are chosen on the validation slice
(newest 20% of the training period, 446 commits) by maximising F1, never on the test set.
Class-imbalance handling for this comparison: **class weights (balanced / scale_pos_weight)**.

## Leaderboard (sorted by mean CV ROC-AUC on the training period)

| # | Model | Type | Tuning | CV ROC-AUC (train) | Test ROC-AUC | Test PR-AUC | F1 | Precision | Recall | MCC | Bal. acc. | Recall@20%Effort |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | XGBoost | Ensemble (boosting) | tuned | 0.732 ± 0.029 | 0.779 | 0.401 | 0.410 | 0.276 | 0.798 | 0.307 | 0.713 | 0.083 |
| 2 | Random Forest | Ensemble (bagging) | tuned | 0.727 ± 0.020 | 0.776 | 0.423 | 0.393 | 0.260 | 0.810 | 0.287 | 0.700 | 0.119 |
| 3 | SVM (linear) | Linear | not tuned | 0.726 ± 0.030 | 0.762 | 0.349 | 0.396 | 0.266 | 0.774 | 0.286 | 0.698 | 0.012 |
| 4 | Logistic Regression | Linear | tuned | 0.725 ± 0.040 | 0.769 | 0.362 | 0.386 | 0.270 | 0.679 | 0.261 | 0.677 | 0.012 |
| 5 | AdaBoost | Ensemble (boosting) | not tuned | 0.725 ± 0.028 | 0.783 | 0.389 | 0.418 | 0.298 | 0.702 | 0.306 | 0.705 | 0.083 |
| 6 | SVM (RBF) | Kernel | tuned | 0.724 ± 0.042 | 0.768 | 0.358 | 0.397 | 0.259 | 0.845 | 0.298 | 0.708 | 0.060 |
| 7 | MLP (neural network) | Neural network | not tuned | 0.708 ± 0.038 | 0.751 | 0.339 | 0.374 | 0.291 | 0.524 | 0.240 | 0.649 | 0.000 |
| 8 | Naive Bayes | Probabilistic | not tuned | 0.700 ± 0.024 | 0.707 | 0.316 | 0.385 | 0.269 | 0.679 | 0.259 | 0.676 | 0.012 |
| 9 | KNN | Instance-based | not tuned | 0.673 ± 0.060 | 0.715 | 0.296 | 0.356 | 0.276 | 0.500 | 0.215 | 0.634 | 0.190 |
| 10 | Decision Tree (ID3-style, entropy) | Tree | not tuned | 0.664 ± 0.028 | 0.766 | 0.358 | 0.383 | 0.248 | 0.833 | 0.276 | 0.693 | 0.119 |
| 11 | Decision Tree (CART, gini) | Tree | not tuned | 0.630 ± 0.027 | 0.750 | 0.351 | 0.347 | 0.215 | 0.905 | 0.236 | 0.659 | 0.119 |
| 12 | Perceptron | Linear | not tuned | 0.628 ± 0.040 | 0.523 | 0.188 | 0.237 | 0.155 | 0.500 | 0.012 | 0.508 | 0.131 |
| 13 | One-Class SVM (anomaly baseline) | Anomaly detection | not tuned | 0.605 ± 0.040 | 0.662 | 0.258 | 0.331 | 0.233 | 0.571 | 0.176 | 0.619 | 0.000 |

## Selected model: XGBoost

Chosen because it has the highest mean cross-validated ROC-AUC on the training period
(0.732 ± 0.029); test scores played no part in the choice.
Decision threshold 0.514 (max F1 on the validation slice). Hyperparameters: `{"colsample_bytree": 0.6481, "learning_rate": 0.0175, "max_depth": 2, "min_child_weight": 7, "n_estimators": 108, "reg_lambda": 3.5034, "subsample": 0.6795}`.

Models whose CV ROC-AUC is within one standard deviation of the winner: XGBoost, Random Forest, SVM (linear), Logistic Regression, AdaBoost, SVM (RBF), MLP (neural network).
These are practically tied, so simplicity, speed and explainability are fair tie-breakers.

Test set: ROC-AUC 0.779, PR-AUC 0.401 (random = 0.150),
F1 0.410, precision 0.276, recall 0.798,
MCC 0.307, balanced accuracy 0.713,
Recall@20%Effort 0.083.

## Hyperparameter optimisation (RandomizedSearchCV, TimeSeriesSplit, scoring = ROC-AUC)

| Model | CV ROC-AUC (defaults) | CV ROC-AUC (after search) | Result | Best hyperparameters |
|---|---|---|---|---|
| XGBoost | 0.673 | 0.732 | tuned | `{"colsample_bytree": 0.6481, "learning_rate": 0.0175, "max_depth": 2, "min_child_weight": 7, "n_estimators": 108, "reg_lambda": 3.5034, "subsample": 0.6795}` |
| Random Forest | 0.705 | 0.727 | tuned | `{"n_estimators": 300, "min_samples_leaf": 20, "max_features": 1.0, "max_depth": 16}` |
| Logistic Regression | 0.721 | 0.725 | tuned | `{"C": 0.0286}` |
| SVM (RBF) | 0.681 | 0.724 | tuned | `{"C": 6.8479, "gamma": 0.0026}` |

## Class imbalance (default hyperparameters, only the imbalance handling changes)

| Model | Imbalance handling | CV ROC-AUC | CV PR-AUC | Test ROC-AUC | Test PR-AUC | F1 (tuned thr.) | Recall (tuned thr.) | Recall @0.5 | F1 @0.5 |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | no handling | 0.721 | 0.437 | 0.760 | 0.346 | 0.392 | 0.738 | 0.310 | 0.349 |
| Logistic Regression | class weights (balanced / scale_pos_weight) | 0.721 | 0.436 | 0.761 | 0.348 | 0.389 | 0.786 | 0.738 | 0.397 |
| Logistic Regression | SMOTE oversampling (training folds only) | 0.719 | 0.433 | 0.758 | 0.341 | 0.416 | 0.738 | 0.738 | 0.413 |
| Random Forest | no handling | 0.698 | 0.399 | 0.734 | 0.393 | 0.367 | 0.774 | 0.286 | 0.366 |
| Random Forest | class weights (balanced / scale_pos_weight) | 0.705 | 0.392 | 0.736 | 0.381 | 0.365 | 0.798 | 0.667 | 0.377 |
| Random Forest | SMOTE oversampling (training folds only) | 0.691 | 0.374 | 0.733 | 0.355 | 0.350 | 0.881 | 0.560 | 0.392 |
| XGBoost | no handling | 0.684 | 0.378 | 0.732 | 0.352 | 0.296 | 0.869 | 0.452 | 0.427 |
| XGBoost | class weights (balanced / scale_pos_weight) | 0.673 | 0.366 | 0.731 | 0.352 | 0.304 | 0.940 | 0.548 | 0.374 |
| XGBoost | SMOTE oversampling (training folds only) | 0.678 | 0.382 | 0.740 | 0.383 | 0.287 | 0.940 | 0.512 | 0.400 |

With the threshold tuned on validation data, the three strategies end up close; the difference shows at the
naive 0.5 threshold, where an unweighted model flags far fewer buggy commits.

## Bias-variance trade-off

* Decision tree complexity curve (`figures/model_complexity_curve.png`): as max_depth grows, training ROC-AUC
  keeps rising (bias falls) while validation ROC-AUC peaks at max_depth = 1 (0.674) and then
  drops (variance grows). At depth 15 the tree scores 0.997 on its training folds but only
  0.518 on validation folds: it memorises the training commits (overfitting).
* Learning curve of the selected model: `figures/model_learning_curve.png`.

## Notes and caveats

* PR-AUC of random guessing equals the test bug rate (0.150), so compare PR-AUC against that.
* For tuned models the CV score is the best of 25 random candidates, so it is slightly optimistic
  (nested cross-validation would remove this bias but costs 5x more time).
* The One-Class SVM never sees buggy commits; it is an anomaly-detection baseline and cannot be selected.
* Recall@20%Effort rewards models that rank *small* risky commits first. Big changes are risky but use up
  the review budget quickly, so a model can have a good ROC-AUC and a low Recall@20%Effort.
* Labels come from the SZZ algorithm, which is noisy: some "clean" commits contain bugs nobody has fixed yet.

## Metric glossary

* ROC-AUC: chance that a random buggy commit gets a higher risk score than a random clean one (0.5 = guessing).
* PR-AUC (average precision): area under the precision-recall curve; guessing scores the bug rate.
* F1: harmonic mean of precision and recall at the chosen threshold.
* Precision: of the commits we flag as risky, the fraction that really are buggy.
* Recall: of the truly buggy commits, the fraction we flag.
* MCC: correlation between predictions and truth, -1..1; stays honest on imbalanced data.
* Balanced accuracy: average of recall on buggy commits and recall on clean commits.
* Recall@20%Effort: share of buggy commits found when reviewing the riskiest commits first until 20% of all changed lines (la + ld) have been read.
