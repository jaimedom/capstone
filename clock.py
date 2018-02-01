import pandas as pd
import datetime
import pytz
import re
from apscheduler.schedulers.blocking import BlockingScheduler

sched = BlockingScheduler()

@sched.scheduled_job('cron', day_of_week='mon-sun', hour=17)
def scheduled_job():
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

    df.sort_values(by='county', inplace=True)
    df.reset_index(inplace=True,drop=True)
    df.to_csv('map.csv', index=False)
    
sched.start()