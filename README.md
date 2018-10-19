# Credit Card Fraud

The data analysed in these notebooks can be found at https://www.kaggle.com/mlg-ulb/creditcardfraud

The idea of the analysis performed in these notebooks is:

1. To emphasize the importance of having good features in problems with strongly imbalanced classes
2. To illustrate how oversampling strategies do not improve the performance of a machine learning model, and in some cases can degrade it.
3. To illustrate that metrics other than accuracy should be reported when evaluating the performance of a ML model on an strongly imbalanced dataset

Four models were built to come up with fraud prediction: 

1. Random forests
2. Random forests with SMOTE upsampling
3. Logistic regression
4. Logistic regression with SMOTE upsampling.

In the first two models the introduction of SMOTE upsampling did very little in the way of improving the performance of the model: ROC AUC went from 0.958 to 0.977 when using SMOTE. Recall and precision went from (0.76, 0.93) to (0.80, 0.85)

The logistic regression model experimented a degradation in performance when using SMOTE: ROC AUC went from 0.975 to 0.974. More importantly, the use of SMOTW upsampling brough recall and precision from (0.63, 0.86) to (0.93, 0.06) for the pre-defined 0.5 decision threshold. 

Bottom line: when working with strongly imbalanced datasets, focus on getting good features!
