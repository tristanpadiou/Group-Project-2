import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score,accuracy_score
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler

# data cleaning function

def cleaning(df):
    
    #df.drop_duplicates(keep='first', inplace=True)
    def quality(rating):
        if rating >= 7:
            return 'promising'
        elif rating <=5:
            return 'cooking wine'
        else:
            return 'trivial wine'
    df['rating'] = (df['quality']).apply(quality)
    return df

def training(df):
    df=cleaning(df)

    X = df.drop(columns=['quality','rating'])
    y = df['rating']

    # encoding

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=1)
    
    # resampling: creating an automation to pick the best sampling method
    samplers=[RandomOverSampler(random_state=1),SMOTE(random_state=1, sampling_strategy='auto'),SMOTEENN(random_state=1),ClusterCentroids(random_state=1),RandomUnderSampler(random_state=1)]
    Scores=[]
    for sampler in samplers:
        sample=sampler
        X_resampled, y_resampled = sample.fit_resample(X_train_orig, y_train_orig)
        cc_model = RandomForestClassifier()
        cc_model.fit(X_resampled, y_resampled)
        cc_y_pred = cc_model.predict(X_test_orig)
        Scores.append([
            accuracy_score(cc_y_pred,y_test_orig),
            sampler
        ])
    sorted_scores = sorted(Scores, reverse=True)
    best_resampler = sorted_scores[0][1]
    print(f'resampling method used: {best_resampler}')
    # resampling with best resampler
    resampler=best_resampler
    X_resampled, y_resampled = resampler.fit_resample(X_train_orig, y_train_orig)


    # hyperparameter tuning
    ## Running the model first
    cc_model_tunning = RandomForestClassifier()
    cc_model_tunning.fit(X_resampled, y_resampled)
    #setting up gridsearch
    param_grid={
    'n_estimators':[50,100,200],
    'max_depth':[None, 1,2,3,4,5,6,7,8,9,10,20,30]}
    grid_search=GridSearchCV(estimator=cc_model_tunning,param_grid=param_grid, cv=5)
    grid_search.fit(X_resampled,y_resampled)

    #   training
    cc_model = RandomForestClassifier()
    #   setting the hyperparameters as per the gridsearch
    cc_model.set_params(**grid_search.best_params_)
    cc_model.fit(X_resampled, y_resampled)
    cc_y_pred = cc_model.predict(X_test_orig)

    print('accuracy_score of Random Forest Classifier:')
    print(accuracy_score(y_test_orig, cc_y_pred))
    print('Classification report of Random Forest Classifier:')
    print(classification_report(y_test_orig, cc_y_pred, target_names=le.classes_))