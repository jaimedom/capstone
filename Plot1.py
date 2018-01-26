from bokeh.sampledata.us_counties import data as counties
from sklearn.externals import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from Transformers import ColumnSelectTransformer, DayProcessor, MonthProcessor, CountyDicGenerator, ThresholdEstimator
from tqdm import tqdm
import datetime
import pytz
import re
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper

# Import predictive model

model = joblib.load('model.pkl')

# Import bokeh areas

counties = {
    code: county for code, county in counties.items() if county["state"] == "ca"
}

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

for i in tqdm(airports.keys()):
    
    temp = {}
    url = url_base+airports[i][0]+'/'+str(year_temp)+'/'+str(month_temp)+'/'+str(day_temp)+'/DailyHistory.html'
    
    table = pd.read_html(url)[0].iloc[:,range(2)]
    table.columns= ['a','b']
    table.set_index('a',drop=True,inplace=True)
    
    temp['county'] = [i]
    temp['startDate'] = [str(month)+'/'+str(day)+'/'+str(year)]
    temp['meanTemp']= [int(re.sub(r'°F','',table['b']['Mean Temperature']).strip())]
    temp['maxTemp']= [int(re.sub(r'°F','',table['b']['Max Temperature']).strip())]
    temp['minTemp']= [int(re.sub(r'°F','',table['b']['Min Temperature']).strip())]
    temp['dewPoint']= [int(re.sub(r'°F','',table['b']['Dew Point']).strip())]
    temp['avgHumidity']= [int(table['b']['Average Humidity'])]
    temp['maxHumidity']= [int(table['b']['Maximum Humidity'])]
    temp['minHumidity']= [int(table['b']['Minimum Humidity'])]
    temp['maxWind']= [int(re.sub(r'mph','',table['b']['Max Wind Speed']).strip())]
    
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

show(p)