from flask import Flask, render_template, request, redirect
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

app = Flask(__name__)

# Create sklearn classes

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

# Define terms dictionary
    
term ={
        'Cost':'cost',
        'Acres':'burnt_acres',
        'Injuries':'injuries',
        'Destroyed structures':'destroyedStructures',
        'Thretened structures':'threatenedStructures',
        'Airtankers':'airtankers',
        'Bulldozers':'dozers',
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
    
    # Obtain input
    
    selected_county = request.args.get('selected_county')
    if selected_county == None:
        selected_county = 'Alameda'
        
    selected_x = request.args.get('selected_x')
    if selected_x == None:
        selected_x = 'Mean temperature'
        
    selected_y = request.args.get('selected_y')
    if selected_y == None:
        selected_y = 'Acres'
    
    # Create the plot
    
    x = list(df[df['county']==selected_county][term[selected_x]]) 
    y = list(df[df['county']==selected_county][term[selected_y]])
    
    # Get rid of missing values
    
    x_ind = [a == a for a in x]
    y_ind = [b == b for b in y]
    index_clean = []
    
    for k in range(len(x_ind)):
        
        index_clean.append(x_ind[k]*y_ind[k])
        
    x = [i for indx,i in enumerate(x) if index_clean[indx]]
    y = [j for indy,j in enumerate(y) if index_clean[indy]]
    
    source = ColumnDataSource(data=dict(x=x, y=y))
    
    plot = figure(plot_height=400, plot_width=400, 
                  title=selected_x+' vs '+selected_y+' in '+selected_county+' County',
                  tools="pan,reset,save,wheel_zoom")
    
    plot.circle('x', 'y', size=8, source=source)
    
    plot.xaxis.axis_label = selected_x
    plot.yaxis.axis_label = selected_y
    
    # Create the widgets    
    
    script, div = components(plot)
   
    return render_template("historic.html", 
                           script=script, 
                           div=div,
                           selected_county=selected_county,
                           selected_x=selected_x,
                           selected_y=selected_y,
                           county_names=sorted(list(set(df['county']))),
                           axis_options = sorted(list(term.keys())))


@app.route('/map')
def map():
    
    # Import bokeh areas
    
    with open('counties', 'rb') as in_strm:
        counties = dill.load(in_strm)
    
    # Generate current values
    
    df = pd.read_csv('map.csv')
    df.startDate = pd.to_datetime(df.startDate)
        
    # Inputs for the plot
    
    county = list(df['county'])
    predicted = model.predict(df)
    
    # Generate plot
    
    county_xs = [c["lons"] for c in counties.values()]
    county_ys = [c["lats"] for c in counties.values()]
    county_cs = [c["name"] for c in counties.values()]
    
    indexes = [county.index(c) for c in county_cs]
    risk = [predicted[i] for i in indexes]
    
    color_mapper = CategoricalColorMapper(palette=["red", "green"], factors=[True, False])
    
    source = ColumnDataSource(data=dict(
        x=county_xs,
        y=county_ys,
        name=county_cs,
        risk=risk,
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
    
@app.route('/local')
def local():
     
    airports = pd.read_csv('airports.csv').set_index('county').T.to_dict('list')    
    counties = sorted(list(airports.keys()))
    
    return render_template("local.html", counties=counties)

@app.route('/forecast')    
def forecast():
    
    try:
        df_pred = pd.DataFrame.from_dict({
                'avgHumidity' : [float(request.args.get('avgHumidity'))],
                'county' : [request.args.get('county_local')],
                'dewPoint' : [float(request.args.get('dewPoint'))],
                'maxHumidity' : [float(request.args.get('maxHumidity'))],
                'maxTemp' : [float(request.args.get('maxTemp'))],
                'maxWind' : [float(request.args.get('maxWind'))],
                'meanTemp' : [float(request.args.get('meanTemp'))],
                'minHumidity' : [float(request.args.get('minHumidity'))],
                'minTemp' : [float(request.args.get('minTemp'))],
                'startDate' : [request.args.get('startDate')]
                })
    except:
        
        return redirect('/wrong')

    if df_pred.shape[1] != 10 or df_pred.avgHumidity[0]<0 or df_pred.avgHumidity[0]>100 or df_pred.maxHumidity[0]<0 or df_pred.maxHumidity[0]>100 or df_pred.minHumidity[0]<0 or df_pred.minHumidity[0]>100:
        
        return redirect('/wrong')
    
    else:
        
        df_pred.startDate = pd.to_datetime(df_pred.startDate)
        predicted_local = model.predict(df_pred)

        if predicted_local[0]:

            return redirect('/risk')

        else:

            return redirect('/safe')
    
@app.route('/safe')    
def safe():
    
    return render_template("safe.html")

@app.route('/risk')    
def risk(): 
    
    return render_template("risk.html")

@app.route('/wrong')    
def wrong(): 
    
    return render_template("wrong.html")

@app.route('/map_test')
def map_test():
    
    # Import bokeh areas
    
    with open('counties', 'rb') as in_strm:
        counties = dill.load(in_strm)
    
    # Generate current values
    
    df = pd.read_csv('map_test.csv')
    df.startDate = pd.to_datetime(df.startDate)
        
    # Inputs for the plot
    
    county = list(df['county'])
    predicted = model.predict(df)
    
    # Generate plot
    
    county_xs = [c["lons"] for c in counties.values()]
    county_ys = [c["lats"] for c in counties.values()]
    county_cs = [c["name"] for c in counties.values()]
    
    indexes = [county.index(c) for c in county_cs]
    risk = [predicted[i] for i in indexes]
    
    color_mapper = CategoricalColorMapper(palette=["red", "green"], factors=[True, False])
    
    source = ColumnDataSource(data=dict(
        x=county_xs,
        y=county_ys,
        name=county_cs,
        risk=risk,
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

