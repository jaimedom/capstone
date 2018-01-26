from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas as pd
from sklearn import base
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from tqdm import tqdm
from sklearn import model_selection
from sklearn.utils import shuffle
import random
from sklearn.externals import joblib
from datetime import datetime

# Import the fire related data

df_true = pd.read_csv('fires_final.csv', parse_dates = ['startDate'])
df_false = pd.read_csv('false_data.csv', parse_dates = ['startDate'])

df = df_false.append(df_true[df_false.columns], ignore_index=True)

holidays = USFederalHolidayCalendar().holidays(start=datetime(1969,12,31), 
                                               end=datetime(2100,12,31)
                                              )

# Clean data before building machine learning
# Asume that there is no wind if not reported

df.fillna(value = {'maxWind': 0}, inplace = True)

# Eliminate the rows with missing values

df.dropna(inplace=True)
df.describe()

# Define classes for machine learning

class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, colnames):
        
        self.cols = colnames
        
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X):
        
        return np.array(X[self.cols].values.tolist())

  
class DayProcessor(base.BaseEstimator, base.TransformerMixin):
    
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X):
        
        dates = pd.DatetimeIndex([i[0] for i in X])
        
        holidays = USFederalHolidayCalendar().holidays(start=datetime(1969,12,31), 
                                                       end=datetime(2100,12,31))
        
        a = np.array([x in holidays for x in list(dates)])
        b = np.array([x.dayofweek in [5,6] for x in list(dates)])
        
        return np.stack((a,b), axis=1)
    
class MonthProcessor(base.BaseEstimator, base.TransformerMixin):
    
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X):
        
        dates = pd.DatetimeIndex([i[0] for i in X])
        
        return [{x.month: 1} for x in list(dates)]
    
class CountyDicGenerator(base.BaseEstimator, base.TransformerMixin):
    
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X):
        
        return [{''.join(x): 1} for x in X.tolist()]
    
class ThresholdEstimator(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self, model, r):
        
        self.model = model
        self.r = r
    
    def fit(self, X, y):
        
        self.model.fit(X,y)
        
    def predict(self, X):
        
        return [True if k[1] > self.r else False for k in self.model.predict_proba(X)]
    
day_features = Pipeline([
                         ('date',ColumnSelectTransformer(['startDate'])),
                         ('day',DayProcessor())
                        ])

month_vectorizer = Pipeline([
                             ('data',ColumnSelectTransformer(['startDate'])),
                             ('month',MonthProcessor()),
                             ('vectorizer', DictVectorizer(sparse = False))
                            ])

    
county_vectorizer = Pipeline([
                              ('county',ColumnSelectTransformer(['county'])),
                              ('dict', CountyDicGenerator()),
                              ('vectorizer', DictVectorizer(sparse = False))
                             ])

weather_variables = ['avgHumidity', 'dewPoint', 'maxHumidity', 'maxTemp',
                     'maxWind', 'meanTemp', 'minHumidity', 'minTemp']

label = np.array(df['fire'])

features = FeatureUnion([
        ('date',day_features),
        ('month',month_vectorizer),
        ('county',county_vectorizer),
        ('weather',ColumnSelectTransformer(weather_variables))
    ])

# Define base models

model1 = Pipeline([
                  ('features',features),
                  ('linear',linear_model.LogisticRegression())
                 ])


model2 = Pipeline([
                  ('features',features),
                  ('svm',svm.SVC(C=1, kernel='linear', probability=True))
                 ])

model3 = Pipeline([
                  ('features',features),
                  ('naive',GaussianNB())
                 ])

model4 = Pipeline([
                  ('features',features),
                  ('tree',DecisionTreeClassifier(min_samples_leaf = 1))
                 ])

# Select optimum hyperparameters for the models

random.seed(100)
df_shuffle = shuffle(df)
label_shuffle = np.array(df_shuffle['fire'])

# Model 1
                  
auc1 = []


for i in tqdm([10**x for x in range(-4,5)]):
    
    #Test model
    test1 = linear_model.LogisticRegression(C = i)
    
    #Cross validate
    cv_test_auc1 = model_selection.cross_val_score(
                             test1,
                             features.fit_transform(df_shuffle), 
                             label_shuffle, 
                             cv=20,  
                             scoring='average_precision'
                                                    )
    
    auc1.append((cv_test_auc1.mean(),i))

# Set the optimum
    
model1.steps[1] = ('linear',linear_model.LogisticRegression(C = max(auc1)[1]))

# Model 2

auc2 = []

for i in tqdm([10**x for x in range(-4,0)]):
    
    #Test model
    test2 = svm.SVC(C=i, kernel='linear')
    
    #Cross validate
    cv_test_auc1 = model_selection.cross_val_score(
                             test2,
                             features.fit_transform(df_shuffle), 
                             label_shuffle, 
                             cv=20,  
                             scoring='average_precision'
                                                    )
    
    auc2.append((cv_test_auc1.mean(),i))

# Set the optimum
    
model2.steps[1] = ('svm',svm.SVC(C=max(auc2)[1], kernel='linear', probability=True))

# Model 4

auc4 = []

for i in tqdm(range(1,40)):
    
    #Test model
    test4 = DecisionTreeClassifier(min_samples_leaf = i)
    
    #Cross validate
    cv_test_auc4 = model_selection.cross_val_score(
                             test4,
                             features.fit_transform(df_shuffle), 
                             label_shuffle, 
                             cv=20,  
                             scoring='average_precision'
                                                    )
    
    auc4.append((cv_test_auc4.mean(),i))

# Set the optimum
    
model4.steps[1] = ('tree',DecisionTreeClassifier(min_samples_leaf = max(auc4)[1]))

# Obtain a table with the accuracy with different thresholds

results = []

for r in tqdm(range(1,31)):
    
    pred1 = np.array([x[1]>r/100 for x in model1.fit(df,label).predict_proba(df)])
    pred2 = np.array([x[1]>r/100 for x in model2.fit(df,label).predict_proba(df)])
    pred3 = np.array([x[1]>r/100 for x in model3.fit(df,label).predict_proba(df)])
    pred4 = np.array([x[1]>r/100 for x in model4.fit(df,label).predict_proba(df)])
    
    results.append({'Model':'Logistic', 'Threshold':r, 
                    'Accuracy': metrics.accuracy_score(label, pred1),
                    'Recall': metrics.recall_score(label, pred1),
                    'Precision': metrics.precision_score(label, pred1)
                    })
    
    results.append({'Model':'SVM', 'Threshold':r, 
                    'Accuracy': metrics.accuracy_score(label, pred2),
                    'Recall': metrics.recall_score(label, pred2),
                    'Precision': metrics.precision_score(label, pred2)
                    })
    
    results.append({'Model':'Naive', 'Threshold':r, 
                    'Accuracy': metrics.accuracy_score(label, pred3),
                    'Recall': metrics.recall_score(label, pred3),
                    'Precision': metrics.precision_score(label, pred3)
                    })
    
    results.append({'Model':'Tree', 'Threshold':r, 
                    'Accuracy': metrics.accuracy_score(label, pred4),
                    'Recall': metrics.recall_score(label, pred4),
                    'Precision': metrics.precision_score(label, pred4)
                    })

results_df = pd.DataFrame(results)

#Given the results I choose to use a tree with 32 leafs and 10% threshold

model_final = Pipeline([
                  ('features',features),
                  ('tree',DecisionTreeClassifier(min_samples_leaf = 32))
                 ])
    
estimator = ThresholdEstimator(model_final,0.1)
estimator.fit(df,label)

results = estimator.predict(df)
metrics.confusion_matrix(label, results)

# Save model

joblib.dump(estimator, 'model.pkl', compress = 1)