# AI4I 2020 Predictive Maintenance

A machine learning classification project that predicts machine failure
using the **AI4I 2020 Predictive Maintenance Dataset**.

## Overview

The goal is to identify potential machine failures from
operating-condition data. The project follows a supervised learning
workflow, from exploratory data analysis to model comparison and final
test-set evaluation.

## Dataset

-   **Samples:** 10,000
-   **Original input features:** 6
-   **Engineered features:** 3 additional features
-   **Target:** `Machine failure` (0 = No Failure, 1 = Failure)
-   **Class distribution:** 96.61% No Failure / 3.39% Failure
-   **Missing values:** None detected

The identifier columns and failure-mode labels are excluded from model
inputs.

**Source:** [AI4I 2020 Predictive Maintenance Dataset --- UCI
Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

## Methodology

1.  **EDA:** Inspect data structure, feature distributions, missing
    values, and class imbalance.
2.  **Feature Engineering:** Create three features:
    -   Temperature Difference
    -   Mechanical Power
    -   Wear--Torque Interaction
3.  **Preprocessing:** Use a Scikit-learn pipeline with `StandardScaler`
    for numerical features and `OrdinalEncoder` for `Type`.
4.  **Baseline:** Evaluate a `DummyClassifier`.
5.  **Model Comparison:** Compare Logistic Regression, Random Forest,
    SVM, Gradient Boosting, and AdaBoost using 5-fold Stratified
    Cross-Validation.
6.  **Threshold Analysis:** Select probability thresholds based on
    F1-score for the leading models.
7.  **Final Evaluation:** Train the selected model on the training set
    and evaluate it on the held-out test set.

## Cross-Validation Results

Results below are from out-of-fold predictions on the training set,
using engineered features and the F1-maximizing threshold for each
model.

  Model                 Threshold   PR-AUC   F1-score
  ------------------- ----------- -------- ----------
  Random Forest            0.4900   0.8982     0.9000
  Gradient Boosting        0.5473   0.9106     0.8956
  AdaBoost                 0.4558   0.7588     0.8078

**Random Forest** was selected based on its highest F1-score among the
three models examined in threshold analysis, with high precision and
competitive recall.

## Final Test-Set Results

The selected Random Forest model was evaluated on the independent test
set using a probability threshold of **0.49**.

  Metric                          Score
  ---------------------------- --------
  Accuracy                       0.9930
  Precision                      0.9655
  Recall                         0.8235
  F1-score                       0.8889
  PR-AUC (Average Precision)     0.8876
  ROC-AUC                        0.9733

The model detects 56 of the 68 failure cases in the test set. The
results indicate strong classification performance, particularly in
precision, while recall shows that some actual failures remain
undetected.

## Tools

-   Python
-   pandas, NumPy
-   Matplotlib
-   scikit-learn
-   Jupyter Notebook

Install dependencies:

``` bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

## Conclusion

The project compares five classification algorithms and evaluates the
effect of three engineered features. Based on F1-score threshold
analysis, Random Forest is selected as the final model, achieving
**0.8889 F1-score** and **0.8876 PR-AUC** on the independent test set.
