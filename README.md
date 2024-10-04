# Group-Project-2
## Introduction
Welcome to the Wine Quality Analysis project! In this repository, we apply machine learning to predict the factors influencing wine quality. Wine, a beverage cherished globally for its diverse flavors and aromas, is produced through a complex interplay of grape variety, fermentation process, and environmental conditions.
This project leverages machine learning techniques to analyze a dataset containing physicochemical properties of different wines alongside their quality ratings. By employing various predictive models, our goal is to identify key features that contribute to wine quality, enabling producers and enthusiasts alike to make informed decisions.

---

## Overview
Analysis and classification of the quality of wine was performed using machine learning. This README consolidates the key components of our machine learning project, which focuses on classification tasks, dimensionality reduction, model evaluation, and optimization through resampling techniques. The project leverages various machine learning algorithms to build robust models for predicting outcomes based on a given dataset.
## Part 1: Data Preparation and Exploration
Data Loading: The dataset is loaded into a Pandas DataFrame.

Data Cleaning: Missing values are handled, and data types are adjusted for analysis.

Exploratory Data Analysis: Key statistics, distributions, and visualizations are generated to understand the data better.

Feature Engineering: New features may be created based on domain knowledge to enhance model performance.
## Part 2: Data Preprocessing
Label Encoding: Categorical variables are transformed into numerical format using label encoding.

Scaling: Features are standardized or normalized to ensure equal weighting during model training.

Train-Test Split: The dataset is divided into training and testing sets to evaluate model performance.

## Part 3: Model Training and Evaluation
Logistic Regression: Initially applied to the PCA-transformed dataset and original dataset to benchmark performance.

Classification reports are generated for detailed evaluation.

Random Forest Classifier: Implemented with different hyperparameters to assess accuracy and performance.

Utilizes the original dataset to train and predict outcomes, generating classification reports to evaluate precision, recall, and F1-scores.

## Part 4: PCA and Logistic Regression
PCA Transformation: Applied PCA to reduce dimensionality while preserving variance.

Logistic Regression on PCA Data: Model fitting and evaluation yield an accuracy score of 78.81% with a detailed classification report.

Comparison with Original Dataset: Logistic Regression on original features achieved an accuracy of 79.95%, showing comparable results.

For this reason, we didn't use PCA.

## Part 5: Model Optimization with Resampling Techniques
Imbalance Handling: Various sampling techniques from imblearn are implemented to address class imbalance:

    -  Random Over Sampler
  
    -  SMOTE
  
    -  SMOTEENN
  
     - Cluster Centroids
  
     - Random Under Sampler

Resampling and Evaluation:
  For each sampler, the data is resampled, and a Random Forest classifier is trained.
  Classification reports are generated, highlighting performance metrics:
    Best performing method: Random Over Sampler with an accuracy of 81%.

Balanced Accuracy Score: The score for the model trained with SMOTE was recorded as 0.76, indicating improved performance for the minority class. 

In the end, we used SMOTE for its ability to perform well with both white and red wine.

---

# Installation Instructions

1. **Clone or Download the Repository**
   - Clone the repository using Git:
     ```bash
     git clone https://github.com/tristanpadiou/Group-Project-2.git
     ```
   - Or download the project as a zip file from the repository and extract it.

2. **Navigate to the Project Directory**
   ```bash
   cd Group-Project-2
   ```

3. **Install Required Packages**
   - Install the necessary packages using the following command:
     ```bash
     pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn scipy
     ```

4. **Data Preparation**
   - The project includes two wine datasets (`winequality-red.csv` and `winequality-white.csv`), which are located in the `Resources` folder. Ensure they remain in this location for the code to run correctly.

5. **Running the Jupyter Notebooks**
   - Start Jupyter notebooks by executing:
     ```bash
     jupyter notebook
     ```
   - Open the following notebooks and follow the instructions within:
     - `Model_Training.ipynb`
     - `WineRatingDeepAnalysis.ipynb`

6. **Running the Model_training.ipynb**
   - This file makes use of the functions in the Pipeline.py file to train the model on the two datasets with as little code as possible

7. **Results and Output**
   - Both the notebooks and Python script will generate predictions and visualizations based on the provided wine quality data.

---

## Key Libraries Used
Pandas: For data manipulation.

NumPy: For numerical operations.

Scikit-learn: For machine learning algorithms, model evaluation, and metrics.

Imbalanced-learn: For implementing resampling techniques.

## Conclusion
In this project, we aimed to discover whether we could accurately predict the quality of wine based on its measurable chemical properties. Through data exploration and model development we were able to demonstrate that it is indeed possible to classify wine quality using these features. We came to the conclusion that the most important factor for a wine's rating is alcohol content as shown in both red and white wines. However, differences start to appear when we look at the second and third most important aspects. In white wine its density at number two and free sulfur dioxide at three. In contrast, sulfates rank second in importance for red wine, and the volatile acidity is third. This project successfully demonstrates a comprehensive approach to data preparation, exploration, model training, and optimization for classification of wine quality. Our Random Forest Classifier achieved a very good balanced accuracy, showing the model's strong performance. By applying resampling methods such as SMOTE and RandomOverSampler, we enhance the model's ability to predict outcomes accurately, particularly in the presence of class imbalances. The results demonstrated the value of machine learning in predicting wine quality based on measurable chemical properties, and shows a promising path ahead for wine producers to enhance and control the quality of their wine in a much more efficient way.

