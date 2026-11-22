# DIABETES PREDICTION USING MACHINE LEARNING

## Machine Learning Project Report

**Project Title:** Diabetes Prediction Using Logistic Regression  
**Dataset:** Diabetes Health Indicators (BRFSS 2015)  
**Programming Language:** Python  
**Development Environment:** Google Colab  
**Machine Learning Type:** Supervised Learning  
**Problem Type:** Binary Classification  


# Abstract

Diabetes is a chronic health condition that affects millions of people worldwide. Early identification of individuals who may be at risk of diabetes can help support timely medical assessment and preventive healthcare.

This project focuses on developing a Machine Learning classification model to predict diabetes status using health-related indicators. The Diabetes Health Indicators Dataset (BRFSS 2015) was used for model development. The dataset contains 253,680 records and 22 columns, including the target variable and 21 input features.

Python libraries, including Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn, were used for data exploration, preprocessing, visualization, model training, and evaluation.

Logistic Regression was selected as the classification algorithm. The dataset was divided into training and testing sets using an 80:20 ratio. StandardScaler was applied to standardize the input features.

The model achieved an accuracy of 86.21% and an ROC-AUC score of 81.92%. However, its recall was 15.83%, indicating that the model missed a considerable proportion of diabetes-positive cases.

The project demonstrates the application of Machine Learning techniques to healthcare classification while highlighting the importance of class imbalance and appropriate evaluation metrics.

# 1. Introduction

Machine Learning is a branch of Artificial Intelligence that enables computers to identify patterns in data and make predictions without being explicitly programmed for every individual task.

In healthcare, Machine Learning can be used to analyze patient information, identify potential risk factors, and support decision-making.

Diabetes is a medical condition associated with elevated blood glucose levels. Identifying individuals who may be at risk is an important step toward early medical assessment.

The purpose of this project is to develop a binary classification model that predicts whether an individual has diabetes based on different health indicators and demographic characteristics.

The project uses Logistic Regression, a supervised Machine Learning algorithm suitable for binary classification problems.

# 2. Problem Statement

Identifying individuals who may have diabetes using multiple health indicators can be challenging when dealing with large datasets.

Traditional approaches may require extensive manual analysis of patient information.

This project aims to use Machine Learning to analyze health-related indicators and predict whether an individual belongs to the diabetes-positive or diabetes-negative class.

The main challenge is developing a model that not only achieves good overall accuracy but also identifies diabetes-positive cases effectively.

# 3. Project Objectives

The main objectives of this project are:

1. To explore and understand the Diabetes Health Indicators Dataset.
2. To perform exploratory data analysis (EDA).
3. To identify the target variable and relevant input features.
4. To inspect missing values, duplicate records, and data distributions.
5. To preprocess the dataset for Machine Learning.
6. To divide the dataset into training and testing sets.
7. To implement Logistic Regression for diabetes prediction.
8. To evaluate model performance using multiple classification metrics.
9. To visualize model performance using a confusion matrix, ROC curve, and Precision-Recall curve.
10. To identify limitations and potential improvements.

# 4. Dataset Description

## 4.1 Dataset Information

The project uses the Diabetes Health Indicators Dataset based on the Behavioral Risk Factor Surveillance System (BRFSS) 2015.

The dataset contains information about individuals, including health conditions, lifestyle indicators, and demographic characteristics.

| Dataset Property | Description |
|---|---|
| Total Records | 253,680 |
| Total Columns | 22 |
| Input Features | 21 |
| Target Variable | Diabetes_binary |
| Problem Type | Binary Classification |
| Missing Values | 0 |
| Data Types | Float64 |

## 4.2 Target Variable

The target variable is Diabetes_binary.

It contains two possible classes:

- 0: Individual does not have diabetes.
- 1: Individual has diabetes.

## 4.3 Input Features

The dataset includes health and demographic indicators such as:

- High Blood Pressure
- High Cholesterol
- Cholesterol Check
- BMI
- Smoking
- Stroke History
- Heart Disease or Heart Attack
- Physical Activity
- Fruit Consumption
- Vegetable Consumption
- Heavy Alcohol Consumption
- Healthcare Access
- Healthcare Cost Barriers
- General Health
- Mental Health
- Physical Health
- Difficulty Walking
- Sex
- Age
- Education
- Income

These features are used as input variables to train the classification model.

# 5. Tools and Technologies

The following technologies and Python libraries were used:

| Tool / Library | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab | Development environment |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, model training, and evaluation |

# 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the dataset's structure, quality, and distribution.

## 6.1 Importing Libraries and Loading Dataset

Pandas was used to load the CSV dataset into a DataFrame.

The head() function was used to display the first five records, while info() was used to inspect the dataset structure, column names, and data types.

The dataset shape was identified as 253,680 rows and 22 columns.

## 6.2 Dataset Information

The info() output showed that all 22 columns contained 253,680 non-null values.

All columns were represented using the float64 data type.

## 6.3 Missing Values Analysis

The dataset was checked for missing values using the isnull() function.

**Result: 0 missing values.**

Therefore, missing-value imputation was not required for the current dataset.

## 6.4 Duplicate Records

Duplicate records were identified using the duplicated() function.

**Total duplicate records identified: 24,206.**

These records were detected during the initial data inspection. Their removal was not demonstrated in the submitted implementation.

## 6.5 Target Class Distribution

The target variable was analyzed using value_counts() and a count plot.

The results were:

| Diabetes Status | Records | Percentage |
|---|---:|---:|
| No Diabetes (0) | 218,334 | 86.07% |
| Diabetes (1) | 35,346 | 13.93% |
| Total | 253,680 | 100% |

The dataset demonstrates class imbalance because the number of diabetes-negative records is significantly higher than the number of diabetes-positive records.

This imbalance is important when evaluating the classification model.

## 6.6 Statistical Analysis

The describe() function was used to calculate statistical information such as:

- Mean
- Standard deviation
- Minimum
- Maximum
- Quartiles

This analysis helped understand the distribution and range of numerical features.

# 7. Data Preprocessing

Data preprocessing was performed to prepare the dataset for Machine Learning.

## 7.1 Separating Features and Target

The target variable Diabetes_binary was separated from the remaining columns.

- X: Contains 21 independent input features.
- y: Contains the target variable.

## 7.2 Train-Test Split

The dataset was divided into training and testing sets using Scikit-learn's train_test_split() function.

The following configuration was used:

| Parameter | Value |
|---|---|
| Training Percentage | 80% |
| Testing Percentage | 20% |
| Random State | 42 |
| Stratification | Yes |

The resulting dataset sizes were:

| Dataset | Records | Features |
|---|---:|---:|
| Training Set | 202,944 | 21 |
| Testing Set | 50,736 | 21 |

Stratification was used to preserve the approximate class distribution in both subsets.

## 7.3 Feature Scaling

StandardScaler was used to standardize the input features.

Feature scaling helps ensure that numerical features with different measurement ranges do not disproportionately influence the model.

The scaler was fitted on the training data and subsequently applied to the testing data.

# 8. Model Development

## 8.1 Logistic Regression

Logistic Regression was selected as the Machine Learning algorithm.

It is a supervised classification algorithm used to estimate the probability that an observation belongs to a particular class.

For binary classification, Logistic Regression estimates probabilities between 0 and 1 using the sigmoid function.

The predicted probability is then converted into a class prediction using a classification threshold.

## 8.2 Model Training

The LogisticRegression class from Scikit-learn was used to initialize the model.

The maximum number of iterations was set to 1000.

The model was trained using the training dataset through the fit() function.

After training, the predict() function was used to generate predicted classes, while predict_proba() was used to obtain predicted probabilities.

# 9. Model Evaluation

The model was evaluated using multiple classification metrics.

These metrics provide a broader understanding of model performance than accuracy alone.

## 9.1 Confusion Matrix

The confusion matrix summarizes the relationship between actual and predicted classifications.

The resulting confusion matrix was:

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 42,622 | 1,045 |
| Actual 1 | 5,950 | 1,119 |

The results indicate:

- True Negatives (TN): 42,622
- False Positives (FP): 1,045
- False Negatives (FN): 5,950
- True Positives (TP): 1,119

The model correctly classified many diabetes-negative cases but missed a considerable number of diabetes-positive cases.

## 9.2 Evaluation Metrics

The following results were obtained:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 0.862129 |
| Precision | 0.517098 |
| Recall | 0.158297 |
| F1 Score | 0.242391 |
| Specificity | 0.976069 |
| Error Rate | 0.137871 |
| ROC-AUC | 0.819163 |
| Log Loss | 0.322148 |
| Average Precision | 0.394229 |
| Balanced Accuracy | 0.567183 |
| Matthews Correlation Coefficient | 0.230262 |
| Cohen's Kappa | 0.189454 |
| Negative Predictive Value | 0.877501 |

## 9.3 Accuracy

Accuracy measures the proportion of all predictions that were correct.

The model achieved an accuracy of approximately 86.21%.

However, accuracy alone is insufficient because the dataset contains significantly more negative cases than positive cases.

## 9.4 Precision

Precision measures how many predicted positive cases were actually positive.

The model achieved approximately 51.71% precision.

## 9.5 Recall

Recall measures how many actual positive cases were correctly identified.

The model achieved approximately 15.83% recall.

This is a major limitation because the model failed to identify many diabetes-positive individuals.

## 9.6 F1 Score

The F1 Score combines precision and recall into a single metric using their harmonic mean.

The model achieved an F1 Score of approximately 24.24%.

## 9.7 Specificity

Specificity measures how effectively the model identifies negative cases.

The model achieved approximately 97.61% specificity.

## 9.8 ROC-AUC

The ROC-AUC score measures how effectively the model distinguishes between positive and negative classes across different classification thresholds.

The model achieved an ROC-AUC score of approximately 81.92%.

## 9.9 Additional Evaluation Metrics

Average Precision, Balanced Accuracy, Matthews Correlation Coefficient, Cohen's Kappa, Negative Predictive Value, and Log Loss were also calculated to provide additional insights into classification performance.

# 10. Data Visualization

Different visualizations were generated to analyze the dataset and evaluate the trained model.

## 10.1 Diabetes Class Distribution

A count plot was generated to visualize the distribution of diabetes-positive and diabetes-negative records.

The visualization showed a significant difference between the two classes.

## 10.2 Confusion Matrix Heatmap

A heatmap was generated using Seaborn to visualize correct and incorrect predictions.

It showed that the model performed considerably better at identifying negative cases than positive cases.

## 10.3 ROC Curve

The ROC curve was plotted using the predicted probabilities.

The resulting ROC-AUC score was approximately 0.819.

This indicates that the model has useful class-discrimination ability, although its positive-class recall at the default threshold remains low.

## 10.4 Precision-Recall Curve

The Precision-Recall curve was generated to examine the relationship between precision and recall at different classification thresholds.

The Average Precision score was approximately 0.394.

This visualization is particularly relevant because the dataset is imbalanced.

# 11. Results and Discussion

The Logistic Regression model achieved an accuracy of 86.21%, suggesting that it correctly classified a large proportion of the testing observations.

However, a closer examination of the evaluation metrics reveals that the model performs much better on the negative class than the positive class.

The recall score of 15.83% indicates that only a small proportion of actual diabetes-positive cases were correctly identified.

Similarly, the confusion matrix shows 5,950 false negatives compared with 1,119 true positives.

The high specificity score of 97.61% demonstrates that the model is effective at identifying individuals belonging to the negative class.

The ROC-AUC score of 81.92% indicates that the model can distinguish between the two classes reasonably well across different probability thresholds.

Overall, the results highlight the importance of considering multiple evaluation metrics when working with imbalanced healthcare datasets.

# 12. Limitations

The current implementation has several limitations:

1. The dataset contains an imbalanced target distribution.
2. The model has relatively low recall for diabetes-positive cases.
3. Duplicate records were identified but their removal was not demonstrated in the submitted implementation.
4. The implementation primarily focuses on Logistic Regression.
5. Additional feature selection and feature engineering techniques were not demonstrated.
6. Advanced class imbalance handling techniques were not evaluated.
7. Hyperparameter optimization was not demonstrated.
8. Additional classification algorithms have not been compared.

# 13. Future Improvements

Several improvements can be introduced to enhance the project.

## 13.1 Handling Class Imbalance

Techniques such as class weighting and SMOTE can be investigated to improve the identification of minority-class observations.

## 13.2 Model Comparison

The performance of Logistic Regression can be compared with:

- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

## 13.3 Hyperparameter Tuning

GridSearchCV or RandomizedSearchCV can be used to identify better model configurations.

## 13.4 Feature Selection

Feature selection techniques can be applied to identify the most relevant predictors and potentially reduce unnecessary complexity.

## 13.5 Threshold Optimization

Different classification thresholds can be evaluated to investigate improvements in positive-class recall while monitoring precision.

## 13.6 Pipeline Implementation

A Scikit-learn Pipeline can be implemented to organize preprocessing and model training while reducing the risk of data leakage.

# 14. Conclusion

This project demonstrates the development of a Machine Learning model for diabetes prediction using the Diabetes Health Indicators Dataset.

The project involved dataset exploration, basic data quality analysis, feature-target separation, train-test splitting, feature scaling, Logistic Regression training, and model evaluation.

The model achieved 86.21% accuracy and an ROC-AUC score of 81.92%.

However, its recall of 15.83% indicates that the model requires further improvement in identifying diabetes-positive cases.

The findings demonstrate that accuracy should not be considered the only measure of performance, especially when working with imbalanced healthcare datasets.

Future work can focus on class imbalance handling, feature selection, model comparison, hyperparameter tuning, and threshold optimization.

This project provides practical experience in applying Python and Machine Learning techniques to a real-world healthcare classification problem.

**Note:** This model is an academic Machine Learning project and is not intended to replace professional medical diagnosis or clinical assessment.

# 15. References

1. Behavioral Risk Factor Surveillance System (BRFSS), 2015, Centers for Disease Control and Prevention (CDC).
2. Scikit-learn Documentation: https://scikit-learn.org/
3. Pandas Documentation: https://pandas.pydata.org/docs/
4. NumPy Documentation: https://numpy.org/doc/
5. Matplotlib Documentation: https://matplotlib.org/
6. Seaborn Documentation: https://seaborn.pydata.org/
