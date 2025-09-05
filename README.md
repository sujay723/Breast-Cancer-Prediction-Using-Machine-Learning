# Breast Cancer Prediction Using Machine Learning

A Jupyter notebook-based project to predict breast cancer using machine learning models. The repository includes dataset, model artifacts, and visualizations.

## Project Structure

- `Breast Cancer Prediction.ipynb`  
  Jupyter notebook with data preprocessing, exploratory analysis, model training, evaluation, and visualization.

- `breast_cancer.csv`  
  Dataset used for training and evaluation.

- `brest_cancer.pkl`  
  Pickled trained model (e.g., a scikit-learn classifier).

- `PE_breast_cancer.jpeg`  
  Exploratory plot (e.g., pairplot, distribution plot).

- `roc_breast_cancer.jpeg`  
  ROC curve visualization of model performance.

## Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sujay723/Breast-Cancer-Prediction-Using-Machine-Learning.git
   cd Breast-Cancer-Prediction-Using-Machine-Learning

## 🚀 Usage
## Running the Notebook

1.Open Breast Cancer Prediction.ipynb.

2.Execute cells step by step:

- Load dataset

- Preprocess data (handle missing values, scaling, etc.)

- Train ML models

- Evaluate model performance

- Save trained model (.pkl)



## 🧠 Models Used

Logistic Regression

Random Forest Classifier

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Decision Tree Classifier

The best model (based on accuracy, precision, recall, F1, and ROC curve) is highlighted in the notebook.


## 📊 Evaluation Metrics

Accuracy – Overall correctness

Precision – Correct positive predictions

Recall (Sensitivity) – Ability to detect true positives

F1-Score – Balance between precision and recall

ROC Curve & AUC – Overall discriminative power

📈 ROC curve is included: roc_breast_cancer.jpeg


## 🔮 Future Improvements

Hyperparameter tuning with GridSearchCV / RandomizedSearchCV

Feature selection to improve model efficiency

Deep Learning models (e.g., Neural Networks)

Deployment using Flask / FastAPI for web interface

Streamlit app for interactive predictions


## 📖 Acknowledgements
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

Visualization inspiration: ROC & Exploratory Data Analysis

   
