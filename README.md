# Breast Cancer Prediction Model Using Machine Learning

This project implements a machine learning model to predict whether a breast tumor is malignant or benign based on a set of features extracted from digital images of breast masses.

## Dataset Overview
The dataset contains 569 entries and 33 columns. The `diagnosis` column serves as the target variable, indicating whether the tumor is **malignant** (M) or **benign** (B). The features extracted from the images are as follows:

### Dataset Columns
- **id**: Unique identifier for each entry.
- **diagnosis**: Target variable (M = Malignant, B = Benign).
- **radius_mean**: Mean of distances from the center to points on the perimeter.
- **texture_mean**: Standard deviation of gray-scale values.
- **perimeter_mean**: Mean perimeter of the tumor.
- **area_mean**: Mean area of the tumor.
- **smoothness_mean**: Mean of local variation in radius lengths.
- **compactness_mean**: Mean of (Perimeter² / Area) - 1.0.
- **concavity_mean**: Mean severity of concave portions of the contour.
- **concave points_mean**: Mean number of concave points on the tumor boundary.
- **symmetry_mean**: Mean symmetry of the tumor.
- **fractal_dimension_mean**: Mean "coastline approximation" - 1.

Additional statistical variations and worst-case values are included in the following columns:
- `_se` columns (e.g., `radius_se`, `texture_se`) indicate standard error for the respective feature.
- `_worst` columns (e.g., `radius_worst`, `texture_worst`) represent the worst values for the respective feature.

### Missing or Unused Columns
- **Unnamed: 32**: Contains no data and will be dropped during preprocessing.

## Workflow
1. **Data Preprocessing**:
   - Dropping unused columns (e.g., `Unnamed: 32`).
   - Handling missing or duplicate entries.
   - Scaling features for uniformity using standardization.

2. **Exploratory Data Analysis (EDA)**:
   - Visualizing feature distributions for malignant and benign cases.
   - Correlation analysis to identify important features.

3. **Model Development**:
   - Training multiple machine learning models, including:
     - Logistic Regression
     - Random Forest Classifier
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
   - Hyperparameter tuning using GridSearchCV.

4. **Evaluation**:
   - Assessing models using metrics like:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
   - Visualizing results through confusion matrices and ROC curves.

5. **Prediction**:
   - Making predictions on new data to classify tumors as malignant or benign.

## Prerequisites
- Python 3.8+
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

## Installation and How to run
1. Clone this repository:
   ```bash
   1st clone the project
      git clone https://github.com/Annibadakh/Breast-Cancer-Prediction-Model.git
   2nd train model 
      --pyhton train.py
   3rd run project
      --python app.py
   4th Open http://localhost:5000 in browser and enter input values and click on predict
      input values example: -0.23717126, -0.64487029, -0.11382239, -0.57427777, -0.60294971,
        1.0897546 ,  0.91543814,  0.41448279,  0.09311633,  1.78465117,
        2.11520208,  0.28454765, -0.31910982,  0.2980991 ,  0.01968238,
       -0.47096352,  0.45757106,  0.28733283, -0.23125455,  0.26417944,
        0.66325388,  0.12170193,  0.42656325,  0.36885508,  0.02065602,
        1.39513782,  2.0973271 ,  2.01276347,  0.61938913,  2.9421769
   5th get output

