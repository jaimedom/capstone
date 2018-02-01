from bokeh.io import curdoc
from bokeh.layouts import column, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Select
from bokeh.plotting import figure
import pandas as pd

# Read input information

df = pd.read_csv('fires_final.csv')

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

# Create the plot

selected_county = 'Alameda'
selected_x = 'Mean temperature'
selected_y = 'Acres'
x = list(df[df['county']==selected_county][term[selected_x]]) 
y = list(df[df['county']==selected_county][term[selected_y]])

source = ColumnDataSource(data=dict(x=x, y=y))

plot = figure(plot_height=400, plot_width=400, title="Historic fire-related data",
              tools="crosshair,pan,reset,save,wheel_zoom")

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

# Add to document

curdoc().add_root(layout)
curdoc().title = "test"
