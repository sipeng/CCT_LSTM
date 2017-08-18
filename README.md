# CCT_LSTM

A Hierarchical LSTM Model for CCT Churn Prediction (Group behaviors) [wiki](https://wiki.corp.adobe.com/display/adobeDS/A+Hierarchical+LSTM+Model+for+CCT+Churn+Prediction)

- Python 3.5.2
- Keras 2.0.3
- Tensorflow 1.0.1

## Hierarchical LSTM model v.s. Baseline models

The baseline models I use are:
- Single LSTM model: build a contract (group) level LSTM model only
- Neural Network: fully-connected neural network (MLP)
- Logistic Regression

From the performance comparison, we can see the edge of our hierarchical LSTM model.

The high value of AUC under ROC curve and low value of AUC under PR curve is mainly due to **imbalanced data distribution.**

### AUC under ROC and AUC under PR

Here I pick the best performance of each model **based on its AUC_ROC value**, so it is possible that the AUC_PR value is not the best possible value of that model.

|           | Hierarchical  | Single LSTM  | Neural Network (MLP) | Logistic Regression  |
| ----------|:-------------:|:------------:|:--------------------:|:--------------------:|
| AUC_ROC   | 0.907         | 0.863        | 0.777                | 0.752                |
| AUC_PR    | 0.256         | 0.128        | 0.051                | 0.021                |

Some insights:
- Both LSTM models have much better performances than Neural Network and Logistic Regression, meaning that the **temporal effects** are important for accurate prediction
- Hierarchical LSTM model is better than Single LSTM model, which justifies the introduction of individual level LSTM model. The **information gathered at individual level helps better predict** the churn probabilities
- The improvement of performance is larger in the sense of AUC_PR, and **AUC_PR is a better measure in the presence of imbalanced data distribution**

### ROC curves (order: hierarchical model, single LSTM baseline, neural network baseline, logistic regression baseline)

![ROC curve](/previous_models/0801/2/roc_lstm2_monthly.png)
![ROC curve](/previous_models/0802/1/roc_lstm2_monthly_baseline.png)
![ROC curve](/previous_models/0807/1/roc_mlp_monthly_baseline.png)
![ROC curve](/previous_models/0807/1/roc_lr_monthly_baseline.png)

### Precision-Recall curves (order: hierarchical model, single LSTM baseline, neural network baseline, logistic regression baseline)

![PR curve](/previous_models/0801/2/pr_lstm2_monthly.png)
![PR curve](/previous_models/0802/1/pr_lstm2_monthly_baseline.png)
![PR curve](/previous_models/0807/1/pr_mlp_monthly_baseline.png)
![PR curve](/previous_models/0807/1/pr_lr_monthly_baseline.png)
