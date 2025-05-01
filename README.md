# An optimized XGBoost framework for interpretable and accurate liver disease diagnosis via data analysis and feature engineering.

## Description
This project focuses on detecting liver disease using machine learning models. The dataset contains features related to patients' medical data, including laboratory test results and other relevant metrics. The goal is to classify whether a patient has liver disease based on these features, using various machine learning models. Additionally, the study compares performance using feature engineering and non-feature engineering approaches.

## Dataset Information
- **Source**: The dataset is related to liver disease detection containing various features based on patients' medical records.

  
Citation:

The dataset used in this study is sourced from Kaggle. If anyone wants to reproduce it, you should use the following citation:


Citation: Abhi8923shriv, & Shrivastava, A. (2023). Liver disease patient dataset [Data set]. Kaggle. https://www.kaggle.com/datasets/abhi8923shriv/liver-disease-patient-dataset
- **Features**: 
  - Age of the patient
  - Gender of the patient
  - Bilirubin levels (Total Bilirubin, Direct Bilirubin)
  - Enzyme levels (Alkphos, Sgpt, Sgot)
  - Albumin levels
  - A/G Ratio (Albumin and Globulin Ratio)
  - Total Protein levels
  - Result (1 = Disease, 2 = No Disease)
  
- **Data Preprocessing**:
  - Missing values were handled by filling with appropriate values or using the mean.
  - Gender column was encoded using Label Encoding.
  - Shapiro wilk test.
  - Outliers were removed using the Interquartile Range (IQR) method.
  - Remove Insignificant column from dataset .
  - Standardize the dataset.

## Code Information
The code implements the following operations:
1. **Data Loading**: Reads the dataset using Pandas.
2. **Preprocessing**
3. **Feature Engineering**: New features were created, including various ratios and enzyme activities.
4. *Dataset Split
5. Train set balancing using smote
6. **Model Training**: Multiple classifiers were trained, including:
   - Decision Trees
   - Random Forest
   - XGBoost
   - LightGBM
   - SVM
   - KNN
   - MLP
7. **Model Evaluation**: Models were evaluated using accuracy, confusion matrix, ROC curve, AUC score, and other metrics.
8. **Hyperparameter Tuning**: Grid search was used to optimize the models' hyperparameters.
9. **Cross-Validation**: 10-fold cross-validation was performed for model evaluation.

## Usage Instructions
### Google Colab Setup
To run this project on Google Colab, follow these steps:

1. **Upload the Code and Dataset**:
   - First, download the code from this respiratory and upload it in google colab, make sure your dataset is accessible in Google Colab. You can upload the dataset directly by running the following code in a Colab cell:
     ```python
     from google.colab import files
     uploaded = files.upload()
     ```
   - This will prompt you to upload the dataset (e.g., `Liverp.csv`) from your local machine.

2. **Install Dependencies**:
   - Install the required libraries by running this code:
     ```python
     !pip install -r requirements.txt
     ```
   - If you don't have a `requirements.txt` file, you can install libraries individually:
     ```python
     !pip install pandas scikit-learn matplotlib seaborn xgboost lightgbm imbalanced-learn shap
     ```

3. **Load the Dataset**:
   - After uploading the dataset, load it into a pandas DataFrame:
     ```python
     import pandas as pd
     df = pd.read_csv('Liverp.csv', encoding='ISO-8859-1')
     ```

4. **Run the Preprocessing Steps**:
   - Follow the preprocessing steps in the notebook to clean the data, handle missing values, and perform any necessary encoding.

5. **Train the Models**:
   - Use the code provided to train the models, perform feature engineering, and evaluate the performance.

6. **Model Evaluation**:
   - You can generate evaluation metrics such as accuracy, confusion matrix, and AUC score for each model.

7. **Plot Results**:
   - Visualize the performance of models using `matplotlib` or `seaborn` to create plots like the ROC curve or confusion matrix.





Python (3.6+)

Libraries:

Pandas

Scikit-learn

Matplotlib

Seaborn

XGBoost

LightGBM

Imbalanced-learn

SHAP (for model interpretation)

Methodology
Data Processing
Handling Missing Data:

Missing values in the dataset were imputed using the mean for numerical columns and the string 'Unknown' for categorical columns (e.g., Gender).

Label Encoding:

Categorical features, such as "Gender of the patient," were encoded using LabelEncoder from scikit-learn.

Outlier Removal:

Outliers were removed using the Interquartile Range (IQR) method, where values outside of the range [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR] were discarded.

Feature Engineering
New Features:

The following new features were created:

Bilirubin to Albumin ratio: Total Bilirubin / ALB Albumin

Direct Bilirubin to Total Bilirubin ratio: Direct Bilirubin / Total Bilirubin

Total Enzyme Activity: Sgpt + Sgot

Bilirubin Protein Indicator: A binary feature indicating whether Total Bilirubin is greater than 1.0.

Modeling
Model Selection:

We trained several machine learning models to detect liver disease, including:

Decision Trees (DT)

Random Forest (RF)

XGBoost (XGB)

LightGBM (LGBM)

K-Nearest Neighbors (KNN)

Support Vector Classifier (SVC)

Multi-Layer Perceptron (MLP)

Hyperparameter Tuning:

We used GridSearchCV for hyperparameter tuning to optimize model parameters such as max_depth for decision trees and n_estimators for ensemble models.

Evaluation
Metrics:

Models were evaluated based on various metrics:

Accuracy: Percentage of correct predictions.

AUC (Area Under the Curve): Measures the model's ability to distinguish between classes.

Confusion Matrix: A matrix that shows the true positives, true negatives, false positives, and false negatives.

ROC Curve: A graphical representation of a model's diagnostic ability.

Feature Engineering vs. Non-Feature Engineering
In this study, we explore both feature engineering and non-feature engineering approaches. The performance of models was compared on the original dataset and the augmented dataset with engineered features.

Performance Comparison: Feature Engineering vs. Non-Feature Engineering
This study explores both the feature engineering approach and the non-feature engineering approach. Below are the results of experiments comparing the performance on the original and augmented datasets.

Table 1: Model Performance (AUC) for Original and Augmented Datasets
Model
Original
 
Dataset
Augmented
 
Dataset
DT
0.9573
0.9891
ET
1.0000
1.0000
XGB
1.0000
1.0000
LGBM
0.9998
1.0000
RF
1.0000
1.0000
KNN
0.9996
0.9998
MLP
0.9994
1.0000
SVC
0.9282
0.9462
VC
1.0000
1.0000
Model
DT
ET
XGB
LGBM
RF
KNN
MLP
SVC
VC
​
  
Original Dataset
0.9573
1.0000
1.0000
0.9998
1.0000
0.9996
0.9994
0.9282
1.0000
​
  
Augmented Dataset
0.9891
1.0000
1.0000
1.0000
1.0000
0.9998
1.0000
0.9462
1.0000
​
 
​
 
Table 2: Model Performance (Test Accuracy) for Original and Augmented Datasets
Model

  
Original Dataset (%)
85.41
100.00
99.23
98.97
100.00
99.95
99.95
87.08
99.95
​
  
Augmented Dataset (%)
93.62
100.00
100.00
99.97
100.00
99.97
99.97
92.82
99.97
​
 
​
 


License & Contribution Guidelines
License: This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Contribution: Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

Materials & Methods
Computing Infrastructure:
Operating System: Windows 11.

Hardware: Processor: I7 1165G7, RAM:16GB, DPU: MX450 2 GB .

Evaluation Method:
Test Accuracy: Measures how well the model generalizes to unseen data.

AUC Score: Evaluates the model's ability to distinguish between classes.

Confusion Matrix: Analyzes classification errors and successes.

ROC Curve: Assesses the trade-off between sensitivity and specificity.

Assessment Metrics:
Accuracy: Percentage of correct predictions.

AUC: Area under the ROC curve; measures classification performance.

Confusion Matrix: Shows true positives, false positives, true negatives, and false negatives.

Precision, Recall, F1-Score: Additional metrics to evaluate classification performance.

### Example Code for Google Colab
Here's a minimal example of code to load data, preprocess, and train a simple model:

```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv('Liverp.csv', encoding='ISO-8859-1')

# Preprocessing (Fill missing values)
df['Gender of the patient'] = df['Gender of the patient'].fillna('Unknown')

# Encoding Gender (Label Encoding)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Gender of the patient'] = label_encoder.fit_transform(df['Gender of the patient'])

# Split the dataset into features and target
X = df.drop(columns=['Result'])
y = df['Result']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")



