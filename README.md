# Cerebral-Stroke-Prediction-Using-Random-Forest

# A machine learning project that predicts the probability of having a cerebral stroke, deployed as a streamlit web application.

# Overview:
The project includes multiple models to compare on a stroke dataset with a challenging imbalance in the dataset. The final model is deployed using Streamlit that patients can input their data and get a stroke risk prediction and probability.

# Dataset Features:

1. Gender (Male/Female/Other)
2. Age
3. Hypertension ( 0 = No / 1 = Yes)
4. Heart Disease (0 = No, 1 = Yes)
5. Ever married (0 = No, 1 = Yes)
6. Work Type ( Govt Job/ Never Worked/ Private/ Self Employed/ Children)
7. Residence (Rural/Urban)
8. Average Glucose Level
9. BMI (Body Mass Index)
10. Smoking Status (Never Smoked/ Formerly Smoked/ Smokes/ Unknown)


# Project Pipeline:

1. Preprocessing:
   Explored the dataset structure and target distribution.
   Handling missing and redundant values (BMI ---> Mean Imputed, Smoking_Status ---> Missing values replaced with "Unknown").
   Encoding categorical values using 'LabelEncoder'.
   Scaling numerical values using 'StandardScaler'.
   Addressed class imbalance with SMOTE combined with Random Under Sampling.
   Split data into training and test sets (80%/20%).

2. Visualization:
   Visualized data distribution before and after resampling to confirm balance.

3. Model Training and Evaluation:
   Trained and compared these models:
      1. Logistic Regression
      2. SVM
      3. XGBoost
      4. Random Forest Classifier

   Model evaluation was based on:
      1. Accuracy
      2. Precision
      3. Recall
      4. F1-score
      5. Confusion Matrix

4. Deployment:
   1. Final Model (RandomForestClassifier) and fitted StandardScaler was saved using the joblib library.
   2. Built a Streamit web application for prediction
   3. Tested locally with low-risk and high-risk profiles to validate the model

# Structure:

app.py ---> Streamlit Application
RFC_model.pkl ---> Trained RandomForestClassifier Model
scaler_RFC.pkl ---> Fitted StandardScaler
dataset.csv ---> Dataset used in project
RFC_Stroke.ipynb ---> Full analysis, preprocessing, visualization, and training notebook
Requirements.txt ---> Python dependencies
README.md



This application is made for educational purposes. It is not a certified medical diagnosis tool and should not be used to real life healthcare decisions.

Made by : NoorAldeen Faruq Al-Hara.
