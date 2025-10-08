# lung_cancer_prediction

Lung Cancer Prediction using Support Vector Machine (SVM)
Project Overview
This project aims to predict the likelihood of lung cancer based on a survey dataset. The analysis is performed using Python with popular data science libraries such as pandas, scikit-learn, and seaborn in a Google Colab notebook. A Support Vector Machine (SVM) classification model is trained on the dataset to make predictions.

Dataset
The dataset used is survey lung cancer.csv, which contains 309 entries and 16 columns. The features include demographic information (age, gender) and various lifestyle factors and symptoms (smoking, alcohol consumption, coughing, etc.). The target variable is LUNG_CANCER.

Dependencies
The following Python libraries are used in this project:

pandas

numpy

seaborn

matplotlib

scikit-learn

You can install these dependencies using pip:

pip install pandas numpy seaborn matplotlib scikit-learn


Usage
To run this project, you will need a Python environment with the above libraries installed. The code is written to be executed in a Jupyter Notebook or Google Colab environment.

Clone the repository:

git clone [https://github.com/your-username/lung-cancer-prediction.git](https://github.com/your-username/lung-cancer-prediction.git)


Navigate to the project directory:

cd lung-cancer-prediction


Load the dataset and run the notebook:
Open the notebook and execute the cells sequentially to see the data analysis, visualization, model training, and evaluation.

import pandas as pd
data = pd.read_csv('survey lung cancer.csv')


Data Exploration and Visualization
The notebook includes several steps for exploring and visualizing the data:

Data Information: Checking for null values and the data types of each column using data.info() and data.isnull().sum().

Descriptive Statistics: Summary statistics for the numerical columns are generated using data.describe().

Visualizations:

A count plot for the target variable LUNG_CANCER to visualize the class distribution.

Bar plots, line plots, and histograms for features like WHEEZING.

A bar plot to show the relationship between AGE and LUNG_CANCER.

Violin plots to understand the distribution of features like AGE and SMOKING with respect to the target variable.

A joint plot to see the relationship between CHRONIC DISEASE and LUNG_CANCER.

Model Training and Evaluation
A Support Vector Machine (SVM) model is used for the classification task.

Data Preprocessing: Categorical variables (GENDER, LUNG_CANCER) are encoded into numerical format using LabelEncoder. The ANXIETY column was dropped from the features.

Feature Selection:

Features (X): All columns except LUNG_CANCER and GENDER.

Target (y): LUNG_CANCER.

Train-Test Split: The data is split into training and testing sets with a test size of 20%.

Feature Scaling: StandardScaler is used to scale the features.

Model Evaluation: The model's performance is evaluated using:

Accuracy Score: The model achieved an accuracy of approximately 93.55%.

Confusion Matrix: A confusion matrix is generated to assess the model's predictions.

Classification Report: This provides precision, recall, and F1-score for each class.

Results
The SVM model performed well on the test data, achieving an accuracy of 93.55%. The classification report provides a more detailed look at the model's performance on a per-class basis. The training accuracy was about 94.74%.

Accuracy: 0.9354838709677419

Classification Report: 
               precision    recall  f1-score   support

           0       0.25      0.50      0.33         2
           1       0.98      0.95      0.97        60

    accuracy                           0.94        62
   macro avg       0.62      0.72      0.65        62
weighted avg       0.96      0.94      0.95        62


Conclusion
This project successfully demonstrates the use of an SVM classifier to predict lung cancer from survey data. The model shows high accuracy, but the classification report indicates that it struggles with the minority class. Future work could involve using more advanced techniques to handle the class imbalance, such as oversampling (e.g., SMOTE) or using different classification algorithms.
