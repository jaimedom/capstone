import bs4 
import urllib
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import re

# Import the data

df=pd.read_csv('fires_final.csv',parse_dates=['startDate','finishDate'])

# Define months
# Three times more samples in June, July, August andd September

months = [1,2,3,4,5,6,6,6,7,7,7,8,8,8,9,9,9,10,11,12]

# Create the boundaries table 
# Counties without fires were added manually

counties = list(set(df.county))
counties.sort()

boundaries_table = pd.DataFrame({'county':counties,
                                 'minLat':df.groupby('county')['latitude'].min().values,
                                 'maxLat':df.groupby('county')['latitude'].max().values,
                                 'minLon':df.groupby('county')['longitude'].min().values,
                                 'maxLon':df.groupby('county')['longitude'].max().values
                               })

counties.extend(['Imperial','San Francisco','Sierra'])

boundaries_table = boundaries_table.append(pd.DataFrame({'county':['Imperial','San Francisco','Sierra'],
                         'minLat':[32.75,37.71,39.51],
                         'maxLat':[33.39,37.796,39.675],
                         'minLon':[-116.08,-122.498,-120.92],
                         'maxLon':[-114.81,-122.404,-120.028]
                              }))

boundaries_table.set_index('county', inplace=True, drop= True)

# Create the false values data frame

month_index = 0 

url_base = 'http://aviationweather.gov/adds/dataserver_current/httpparam?dataSource=stations&requestType=retrieve&format=xml&radialDistance=40;'
url_base_2 = 'https://www.wunderground.com/history/airport/'

df_false = pd.DataFrame()

df_false['county']=np.nan
df_false['meanTemp']= np.nan
df_false['maxTemp']= np.nan
df_false['minTemp']= np.nan
df_false['dewPoint']= np.nan
df_false['avgHumidity']= np.nan
df_false['maxHumidity']= np.nan
df_false['minHumidity']= np.nan
df_false['maxWind']= np.nan
df_false['fire']= np.nan

control_val = 0

for i in tqdm(counties):
    
    for _ in tqdm(range(int(10*len(df)/len(counties)))):
        
        lon = np.random.uniform(boundaries_table.minLon[i],
                                      boundaries_table.maxLon[i])
        lat = np.random.uniform(boundaries_table.minLat[i],
                                      boundaries_table.maxLat[i])
        
        while(True):
            
            day = np.random.randint(1,29)
            month = months[month_index]
            year = np.random.randint(2010,2018)
            
            if datetime.datetime.strptime(str(year)+'-'+str(month)+'-'+str(day), "%Y-%m-%d").date() not in [j.date() for j in df.loc[df.county==i,'startDate']]:
                break
            
        url = url_base+str(lon)+','+str(lat)

        source = urllib.request.urlopen(url).read()
        soup = bs4.BeautifulSoup(source,"lxml")
        airports = [x.text for x in soup.find_all('station_id')]
        
        for j in airports:
    
            try:
                
                url2 = url_base_2+j+'/'+str(year)+'/'+str(month)+'/'+str(day)+'/DailyHistory.html'
                table = pd.read_html(url2)[0].iloc[:,range(2)]
                table.columns= ['a','b']
                table.set_index('a',drop=True,inplace=True)
                
                if(np.logical_or(table.b['Max Temperature']!=table.b['Max Temperature'],table.b['Max Temperature']=='-')):
                    
                    continue
                
                else:
                    
                    dict_temp={}
                    dict_temp['county']=i
                    dict_temp['startDate']= datetime.datetime.strptime(str(year)+'-'+str(month)+'-'+str(day), "%Y-%m-%d")
                    dict_temp['meanTemp']= table['b']['Mean Temperature']
                    dict_temp['maxTemp']= table['b']['Max Temperature']
                    dict_temp['minTemp']= table['b']['Min Temperature']
                    dict_temp['dewPoint']= table['b']['Dew Point']
                    dict_temp['avgHumidity']= table['b']['Average Humidity']
                    dict_temp['maxHumidity']= table['b']['Maximum Humidity']
                    dict_temp['minHumidity']= table['b']['Minimum Humidity']
                    dict_temp['maxWind']= table['b']['Max Wind Speed']
                    dict_temp['fire']=0
                    
                    df_false = df_false.append([dict_temp])
                    
                    break
                
            except:
                
                continue
            
        month_index += 1
        control_val += 1
        
        if month_index == 20:
            month_index = 0
            
    month_index = 0
    
df_false.reset_index(inplace=True,drop=True)
    
# Clean dew points, temperatures and wind speed

temp_keys = ['meanTemp','maxTemp','minTemp','dewPoint']

for j in tqdm(range(len(df_false))):
    
    for k in temp_keys:
        
        if(df_false[k][j]==r'-'):
            df_false[k][j]=np.nan
            
        if(df_false[k][j]==df_false[k][j]):
            df_false[k][j]= re.sub(r'°F',"",df_false[k][j]).strip()
            
    if(df_false['maxWind'][j]==r'-'):
            df_false['maxWind'][j]=np.nan
            
    if(df_false['maxWind'][j]==df_false['maxWind'][j]):
            df_false['maxWind'][j]= re.sub(r'mph',"",df_false['maxWind'][j]).strip()        

df_false[temp_keys] = df_false[temp_keys].astype('float')
df_false[['avgHumidity', 'dewPoint', 'maxHumidity', 'maxTemp',
         'maxWind', 'meanTemp', 'minHumidity', 'minTemp']] = df_false[['avgHumidity', 'dewPoint', 'maxHumidity', 'maxTemp',
         'maxWind', 'meanTemp', 'minHumidity', 'minTemp']].astype(float)
       
df_false.to_csv('false_data.csv', index = False)        
