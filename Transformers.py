import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn import base

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
        
        holidays = USFederalHolidayCalendar().holidays(start=min(dates), 
                                                       end=max(dates))
        
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
        
        return [True if k[1]>self.r else False for k in self.model.predict_proba(X)]