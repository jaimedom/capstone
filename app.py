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

app = Flask(__name__)

# Create sklearn classes

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
    
# Import predictive model

model = joblib.load('model.pkl')

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
    
    counties = pd.read_pickle('geocounty.pkl')
    
    # Import airports data
    
    airports = pd.read_csv('airports.csv').set_index('county').T.to_dict('list')
    
    # Get date
    
    day = datetime.datetime.now(pytz.timezone('US/Pacific')).day
    month = datetime.datetime.now(pytz.timezone('US/Pacific')).month
    year = datetime.datetime.now(pytz.timezone('US/Pacific')).year
    
    # Create weather temporal
    
    temp_date = datetime.datetime(year, month, day) - datetime.timedelta(1)
    day_temp = temp_date.day
    month_temp = temp_date.month
    year_temp = temp_date.year
    
    # Generate current values
    
    df = pd.DataFrame()
    url_base = 'https://www.wunderground.com/history/airport/'
    
    for i in airports.keys():
        
        temp = {}
        url = url_base+airports[i][0]+'/'+str(year_temp)+'/'+str(month_temp)+'/'+str(day_temp)+'/DailyHistory.html'
        
        table = pd.read_html(url)[0].iloc[:,range(2)]
        table.columns= ['a','b']
        table.set_index('a',drop=True,inplace=True)
        
        temp['county'] = [i]
        temp['startDate'] = [str(month)+'/'+str(day)+'/'+str(year)]
        temp['meanTemp']= [int(re.findall(r'[0-9]+',table['b']['Mean Temperature'])[0])]
        temp['maxTemp']= [int(re.findall(r'[0-9]+',table['b']['Max Temperature'])[0])]
        temp['minTemp']= [int(re.findall(r'[0-9]+',table['b']['Min Temperature'])[0])]
        temp['dewPoint']= [int(re.findall(r'[0-9]+',table['b']['Dew Point'])[0])]
        temp['avgHumidity']= [int(re.findall(r'[0-9]+',table['b']['Average Humidity'])[0])]
        temp['maxHumidity']= [int(re.findall(r'[0-9]+',table['b']['Maximum Humidity'])[0])]
        temp['minHumidity']= [int(re.findall(r'[0-9]+',table['b']['Minimum Humidity'])[0])]
        temp['maxWind']= [int(re.findall(r'[0-9]+',table['b']['Max Wind Speed'])[0])]
        
        if df.empty:
            df = pd.DataFrame.from_dict(temp)
        else:
            df = df.append(pd.DataFrame.from_dict(temp))
            
    df.reset_index(inplace=True,drop=True)
    df.startDate = pd.to_datetime(df.startDate)
        
    # Inputs for the plot
    
    county = list(df['county'])
    predicted = model.predict(df)
    
    # Generate plot
    
    county_xs = [county["lons"] for county in counties.values()]
    county_ys = [county["lats"] for county in counties.values()]
    
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
