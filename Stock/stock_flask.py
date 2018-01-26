from flask import Flask, render_template, request
import simplejson as json
import requests
import pandas as pd
from bokeh.plotting import figure
from bokeh.embed import components

app = Flask(__name__)

price_options = ['Close', 'Adjusted close', 'Open', 'Adjusted Open']

@app.route('/')
def index():
    
    code = request.args.get('code')
    if code == None:
        code = 'GOOG'
        
    price = request.args.get('price_option')
    if price == None:
        price = 'Open'
    
    try:    
        price_search = {'Close': 'close',
                 'Adjusted close': 'adj_close',
                 'Open': 'open',
                 'Adjusted Open': 'adj_open'
                }[price]
    except:
        price_search = 'open'
        
    api_key = '6LqqHDqZYZX4TaPdPnKR'
    
    quandl = 'https://www.quandl.com/api/v3/datatables/WIKI/PRICES.json?date.gte=20170101&date.lt=20171231'
    quandl += '&ticker=' + code
    quandl += '&qopts.columns=date,' + price_search
    quandl += '&api_key=' + api_key
    
    quandl_data = requests.get(quandl)
    stock_load = json.loads(quandl_data.content) 
    df = pd.DataFrame(stock_load['datatable']['data'])

    p = figure(title = 'Price for '+ code,
               x_axis_label = 'Date',
               y_axis_label = 'Price ($)',
               x_axis_type = 'datetime'
               )
    
    if not df.empty:
        df[0] = pd.DatetimeIndex(df[0])
        p.y_range.start = min(df[1])
        p.y_range.end = max(df[1])
        p.line(x=df[0],y=df[1],color='blue',legend=None)
    
    script, div = components(p)
    
    return render_template("plot.html", script=script, div=div,
		price_options=price_options, current_code=code, 
         current_selected_price=price)
    
if __name__ == '__main__':
	app.run(port=5000, debug=True)
