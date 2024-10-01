# Group-Project-2
##Introduction
Welcome to the Wine Quality Analysis project! In this repository, we apply machine learning to predict the factors influencing wine quality. Wine, a beverage cherished globally for its diverse flavors and aromas, is produced through a complex interplay of grape variety, fermentation process, and environmental conditions.
This project leverages machine learning techniques to analyze a dataset containing physicochemical properties of different wines alongside their quality ratings. By employing various predictive models, our goal is to identify key features that contribute to wine quality, enabling producers and enthusiasts alike to make informed decisions.
##Overview
Analysis and classification of the quality of wine was performed using machine learning. This README consolidates the key components of our machine learning project, which focuses on classification tasks, dimensionality reduction, model evaluation, and optimization through resampling techniques. The project leverages various machine learning algorithms to build robust models for predicting outcomes based on a given dataset.
##Part 1: Data Preparation and Exploration
Data Loading: The dataset is loaded into a Pandas DataFrame.
Data Cleaning: Missing values are handled, and data types are adjusted for analysis.
Exploratory Data Analysis: Key statistics, distributions, and visualizations are generated to understand the data better.
Feature Engineering: New features may be created based on domain knowledge to enhance model performance.
##Part 2: Data Preprocessing
Label Encoding: Categorical variables are transformed into numerical format using label encoding.
Scaling: Features are standardized or normalized to ensure equal weighting during model training.
Train-Test Split: The dataset is divided into training and testing sets to evaluate model performance.
##Part 3: Model Training and Evaluation
Logistic Regression: Initially applied to the PCA-transformed dataset and original dataset to benchmark performance.
Accuracy: 78.81% (PCA) and 79.95% (Original).
Classification reports are generated for detailed evaluation.
Random Forest Classifier: Implemented with different hyperparameters to assess accuracy and performance.
Utilizes the original dataset to train and predict outcomes, generating classification reports to evaluate precision, recall, and F1-scores.
##Part 4: PCA and Logistic Regression
PCA Transformation: Applied PCA to reduce dimensionality while preserving variance.
Logistic Regression on PCA Data:
Model fitting and evaluation yield an accuracy score of 78.81% with a detailed classification report.
Comparison with Original Dataset:
Logistic Regression on original features achieved an accuracy of 79.95%, showing comparable results.
##Part 5: Model Optimization with Resampling Techniques
Imbalance Handling: Various sampling techniques from imblearn are implemented to address class imbalance:
Random Over Sampler
SMOTE
SMOTEENN
Cluster Centroids
Random Under Sampler
Resampling and Evaluation:
For each sampler, the data is resampled, and a Random Forest classifier is trained.
Classification reports are generated, highlighting performance metrics:
Best performing method: Random Over Sampler with an accuracy of 81%.
Balanced Accuracy Score: The score for the model trained with SMOTE was recorded as 0.76, indicating improved performance for the minority class.
##Key Libraries Used
Pandas: For data manipulation.
NumPy: For numerical operations.
Scikit-learn: For machine learning algorithms, model evaluation, and metrics.
Imbalanced-learn: For implementing resampling techniques.
##Conclusion
This project successfully demonstrates a comprehensive approach to data preparation, exploration, model training, and optimization for classification of wine quality. By applying techniques such as PCA and various resampling methods, we enhance the model's ability to predict outcomes accurately, particularly in the presence of class imbalances. 
