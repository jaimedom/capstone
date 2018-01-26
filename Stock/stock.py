import requests
import pandas as pd
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import row, widgetbox 
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Button, RadioButtonGroup, TextInput

# Set the parameters

code = 'GOOG'
price = 'close'
api_key = '6LqqHDqZYZX4TaPdPnKR'

# Create quandl code

quandl = 'https://www.quandl.com/api/v3/datatables/WIKI/PRICES.json?date.gte=20170101&date.lt=20171231'
quandl += '&ticker=' + code
quandl += '&qopts.columns=date,' + price 
quandl += '&api_key=' + api_key

# Obtain the data

json_file = requests.get(quandl).json()
df = pd.DataFrame(json_file['datatable']['data'])
df[0] = pd.DatetimeIndex(df[0])
source = ColumnDataSource(data = {'0':df[0],'1':df[1]})

# Create bokeh initial window

p = figure(title = 'Price for '+ code,
           x_axis_label = 'Date',
           y_axis_label = 'Price ($)',
           x_axis_type = 'datetime'
           )

p.y_range.start = min(df[1])
p.y_range.end = max(df[1])

p.line(x='0',y='1',source=source,color='blue',legend=None)

# Create widgetbox

button = Button(label="Update")

radio_button_group = RadioButtonGroup(labels=['Closing price', 'Adjusted closing price', 
                                              'Opening price', 'Adjusted opening price'], 
                                      active=0)

text_input = TextInput(value="GOOG", title="Company:")

widget_box = widgetbox(text_input, radio_button_group, button)

# Create callback

def update():

    code = text_input.value
    
    price = {'0': 'close',
             '1': 'adj_close',
             '2': 'open',
             '3': 'adj_open'
            }[str(radio_button_group.active)]
    
    api_key = '6LqqHDqZYZX4TaPdPnKR'
    
    # Create quandl code
    
    quandl = 'https://www.quandl.com/api/v3/datatables/WIKI/PRICES.json?date.gte=20170101&date.lt=20171231'
    quandl += '&ticker=' + code
    quandl += '&qopts.columns=date,' + price 
    quandl += '&api_key=' + api_key
    
    # Obtain the data
    
    json_file = requests.get(quandl).json()
    temp = pd.DataFrame(json_file['datatable']['data'])
    temp[0] = pd.DatetimeIndex(temp[0])
    source.data = {'0':temp[0],'1':temp[1]}
    
    # Change the figure
    
    p.title.text = 'Price for '+code
    p.y_range.start = min(temp[1])
    p.y_range.end = max(temp[1])
    p.legend.name.text = code
         
button.on_click(update)

# Create layout

layout = row(widget_box,p)

#show(layout)

curdoc().add_root(layout)
