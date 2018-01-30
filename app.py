from flask import Flask, render_template, request
import pandas as pd
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.layouts import column, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.models.widgets import Select
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.tree import DecisionTreeClassifier
import datetime
import pytz
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn import base
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
import dill

app = Flask(__name__)

# Import the fire related data

df_true = pd.read_csv('fires_final.csv', parse_dates = ['startDate'])
df_false = pd.read_csv('false_data.csv', parse_dates = ['startDate'])

df2 = df_false.append(df_true[df_false.columns], ignore_index=True)

holidays = USFederalHolidayCalendar().holidays()

# Clean data before building machine learning
# Asume that there is no wind if not reported

df2.fillna(value = {'maxWind': 0}, inplace = True)

# Eliminate the rows with missing values

df2.dropna(inplace=True)
df2.describe()

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
        
        holidays = USFederalHolidayCalendar().holidays()
        
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

label2 = np.array(df2['fire'])

features = FeatureUnion([
        ('date',day_features),
        ('month',month_vectorizer),
        ('county',county_vectorizer),
        ('weather',ColumnSelectTransformer(weather_variables))
    ])

model_final = Pipeline([
                  ('features',features),
                  ('tree',DecisionTreeClassifier(min_samples_leaf = 32))
                 ])
    
model = ThresholdEstimator(model_final,0.1)
model.fit(df2,label2)

@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/about')
def about():
    
    return render_template('about.html')

@app.route('/historic')
def historic():
   
    # Read input information

    df = pd.read_csv('fires_final.csv')
    
    # Define terms dictionary
    
    term ={'County':'county',
           'Cost':'cost',
           'Acres':'burnt_acres',
           'Injuries':'injuries',
           'Destroyed structures':'destroyedStructures',
           'Thretened structures':'thretenedStructures',
           'Airtankers':'airtankers',
           'Bulldozers':'buldozers',
           'Fire engines':'fireEngines',
           'Fire personnel':'firePersonnel',
           'Firemen':'fireCrews',
           'Helicopters':'helicopters',
           'Water tenders':'waterTenders',
           'Maximum temperature':'maxTemp',
           'Mean temperature':'meanTemp',
           'Minimum temperature':'minTemp',
           'Dew point':'dewPoint',
           'Maximum humidity':'maxHumidity',
           'Mean humidity':'avgHumidity',
           'Minimum humidity':'minHumidity',
           'Maximum wind':'maxWind'
           }
    
    # Create the plot
    
    selected_county = 'Alameda'
    selected_x = 'Mean temperature'
    selected_y = 'Acres'
    x = list(df[df['county']==selected_county][term[selected_x]]) 
    y = list(df[df['county']==selected_county][term[selected_y]])
    
    source = ColumnDataSource(data=dict(x=x, y=y))
    
    plot = figure(plot_height=400, plot_width=400, title="Historic fire-related data",
                  tools="pan,reset,save,wheel_zoom")
    
    plot.circle('x', 'y', source=source)
    
    # Create the widgets
    
    menu1 = sorted(df['county'].unique())
    menu2 = sorted(term.keys())
    
    select1 = Select(title="County:", value="Alameda", options=menu1)
    select2 = Select(title="Variable X:", value="Mean temperature", options=menu2)
    select3 = Select(title="Variable Y:", value="Acres", options=menu2)
    
    # Create callback
    
    def update_data(attrname, old, new):
        
        selected_county = select1.value
        selected_x = select2.value
        selected_y = select3.value
        x = list(df[df['county']==selected_county][term[selected_x]]) 
        y = list(df[df['county']==selected_county][term[selected_y]])
    
        source.data = dict(x=x, y=y)
        
    for w in [select1, select2, select3]:
        w.on_change('value', update_data)
    
    # Create layout
        
    inputs = widgetbox(select1,select2,select3)
    layout = column(inputs, plot)
    
    script, div = components(layout)
   
    return render_template("historic.html", script=script, div=div)

@app.route('/map')
def map():
    
    # Import bokeh areas
    
    with open('counties', 'rb') as in_strm:
        counties = dill.load(in_strm)
    
    # Generate current values
    
    df = pd.read_csv('map.csv')
    df.startDate = pd.to_datetime(df.startDate)
    print len(df)
        
    # Inputs for the plot
    
    county = list(df['county'])
    predicted = model.predict(df)
    
    # Generate plot
    
    county_xs = [county["lons"] for county in counties.values()]
    county_ys = [county["lats"] for county in counties.values()]
    print len(county_xs)
    
    color_mapper = CategoricalColorMapper(palette=["red", "green"], factors=[True, False])
    
    source = ColumnDataSource(data=dict(
        x=county_xs,
        y=county_ys,
        name=county,
        risk=predicted,
    ))
    
    TOOLS = "pan,wheel_zoom,reset,hover,save"
    
    p = figure(
        title="Current risk of fire in California", tools=TOOLS,
        x_axis_location=None, y_axis_location=None
    )
    p.grid.grid_line_color = None
    
    p.patches('x', 'y', source=source,
              fill_color={'field': 'risk', 'transform': color_mapper},
              fill_alpha=0.7, line_color="white", line_width=0.5)
    
    hover = p.select_one(HoverTool)
    hover.point_policy = "follow_mouse"
    hover.tooltips = [
        ("County", "@name")
    ]

    script, div = components(p)
   
    return render_template("map.html", script=script, div=div)
    

    
if __name__ == '__main__':
    
    app.run(port=5000, debug=True)
