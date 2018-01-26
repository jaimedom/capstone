import bs4 
import urllib
import pandas as pd
import re
import geocoder
import numpy as np

# Define the url to scrap the fire data
url_base = 'http://cdfdata.fire.ca.gov/incidents/incidents_details_info?incident_id='

# Create an empty database
df = pd.DataFrame()

# Webscrap the data
for j in range(2000):
    
    print('Progress: '+str(j)+'/2000')

    # Connect to webpage
    url=url_base+str(j)
    source = urllib.request.urlopen(url).read()
    soup = bs4.BeautifulSoup(source,"lxml")
    
    # Extract the data
    try:
        table = soup.find('table',class_='incident_table')
        rows = table.find_all('tr')
    
        content = []
        n = 0 
        tag_info = []
    
        for tr in rows:
            
            td = tr.find_all('td')
            
            if n!=0:
                tag_info = [x.text.strip() for x in td][0:2]
                
            else:
                tag_info = ['Fire_name']
                tag_info.append(td[0].text)
                n+=1
            
            content.append(tag_info)    
    except:
        continue
    
    # Transform list to dictionary
    
    dict_temp = {}
    
    for i in content:
        dict_temp[i[0]] = [i[1]]
    
    # Append data
    
    if len(df)!=0:
        df = df.append(pd.DataFrame(dict_temp))
        
    else:
        df = pd.DataFrame(dict_temp)


# Clean the data
# Initial steps

df.columns = ['burned_containment', 'unit', 'cause',
       'conditions', 'agencies:', 'cost', 'county',
       'startDate', 'estimated_containment:', 'evacuations',
       'fire_name', 'management_team', 'injuries', 'finishDate',
       'location', 'long/lat', 'phoneNumbers', 'roadClosures',
       'destroyedStructures:', 'threatenedStructures', 'airtankers',
       'dozers', 'fireEngines', 'firePersonnel',
       'fireCrews', 'helicopters', 'waterTenders']

df.reset_index(inplace=True,drop=True)

# Clean the location and obtain coordinates

df['latitude']= float('NaN')
df['longitude']= float('NaN')

for i in range(len(df)):
    
    print('Progress: '+str(i)+'/'+str(len(df)))
    if df['location'][i] != df['location'][i]:
        continue
        
    df['location'][i] = re.sub(r'((\d-)?\d mile(s)? )?(\w*[Nn]orth\w*|\w*[Ww]est\w*|\w*[Ee]ast\w*|\w*[Ss]outh\w*)( of)?',
                                "",df['location'][i])+', CA'
    
    try:  
        coordinates = geocoder.google(df['location'][i]).latlng
        df['latitude'][i] = coordinates[0] 
        df['longitude'][i] = coordinates[1]

    except:
        continue

# Set to nan the counties with ambiguous values

df.loc[np.logical_or(df.county=='Coastal Counties',
       np.logical_or(df.county=='Multiple Counties',
       np.logical_or(df.county=='Multiple counties',
                     df.county=='Southern California'
                     ))),
       'county']=np.nan

# If the county is missing and the coordinates are missing then remove 

boolean_drop = np.logical_and(df.county.isnull(),df.latitude.isnull())
index_drop = [i for i,x in enumerate(boolean_drop) if x] 
df.drop(index_drop, inplace=True)  
df.reset_index(inplace=True,drop=True)

# Get counties from coordinates

boolean_fill = df.county.isnull()   
index_fill = [i for i,x in enumerate(boolean_fill) if x] 

for i in index_fill:
    df.loc[i,'county'] = geocoder.google([df.loc[i,'latitude'], 
                                          df.loc[i,'longitude']], method='reverse').county
    
# Remove fires without county
    
boolean_drop = np.logical_or(df.county.isnull(),df.county=="")
index_drop = [i for i,x in enumerate(boolean_drop) if x] 
df.drop(index_drop, inplace=True)  
df.reset_index(inplace=True,drop=True)

# Standardize counties before filling coordinates missing values
# Remove the 'county' part

for i in range(len(df)):
    
    if df['county'][i] != df['county'][i]:
        continue
    
    df['county'][i] = re.sub(r'County',"",df['county'][i]).strip()
    
# Fix the counties

df.loc[np.logical_or(df.county=='Amador / El Dorado Counties',
       np.logical_or(df.county=='Amador and El Dorado Counties',
                     df.county=='Plymouth (Amador) / Big (El Dorado)'
                     )),
       'county']='Amador'

df.loc[df.county=='Contra Costa Fire','county'] = 'Contra Costa'

df.loc[np.logical_or(df.county=='El Dorado ,Placer',
       np.logical_or(df.county=='El Dorado Unit',
                     df.county=='El Dorado/Sacramento'
                     )),
       'county']='El Dorado'
                     
df.loc[np.logical_or(df.county=='Firefighters are battling multiple fires in Monterey & San Luis Obispo Counties',
       np.logical_or(df.county=='Monterey & San Benito',
                     df.county=='Monterey ,San Luis Obispo'
                     )),
       'county']='Monterey'

df.loc[np.logical_or(df.county=='Inyo and Mono',
                     df.county=='Inyo and Mono counties'),
       'county']='Inyo'
                     
df.loc[np.logical_or(df.county=='Kern  / San Luis Obispo',
                     df.county=='Kern /Ventura'),
       'county']='Kern'
                     
df.loc[np.logical_or(df.county=='Lake & Napa counties',
                     df.county=='Lake and Colusa Counties'),
       'county']='Lake'
                     
df.loc[np.logical_or(df.county=='Lassen & Modoc',
                     df.county=='Lassen and Shasta Counties'),
       'county']='Lassen'
      
df.loc[np.logical_or(df.county=='Los Angeles  (near the San Bernardino  border)',
                     df.county=='Los Angeles  / Angeles National Forest'),
       'county']='Los Angeles'
                     
df.loc[np.logical_or(df.county=='Madera & Mariposa Counties',
                     df.county=='Madera and Mariposa Counties'),
       'county']='Madera'
                     
df.loc[np.logical_or(df.county=='Modac',
                     df.county=='Modoc  & State of Oregon'),
       'county']='Modoc'
                     
df.loc[np.logical_or(df.county=='Napa & Lake Counties',
                     df.county=='Napa ,Sonoma'),
       'county']='Napa'
                     
df.loc[np.logical_or(df.county=='Orange /Riverside',
                     df.county=='Orange/Riverside Counties'),
       'county']='Orange'
                     
df.loc[df.county=='Riverside, CA','county'] = 'Riverside'

df.loc[df.county=='San Benito & Monterey Counties','county'] = 'San Benito'

df.loc[df.county=='San Diego / Riverside','county'] = 'San Diego'

df.loc[df.county=='Santa Clara and Santa Cruz Counties','county'] = 'Santa Clara'

df.loc[np.logical_or(df.county=='Shasta  / Lassen',
                     df.county=='Shasta and Trinity'),
       'county']='Shasta'
                     
df.loc[df.county=='Siskiyou  (Jackson & Klamath counties in Oregon)','county'] = 'Siskiyou'

df.loc[df.county=='Sonoma, Solano, Lake & Colusa Counties','county'] = 'Sonoma'
                     
df.loc[df.county=='Tehama & Shasta Counties','county'] = 'Tehama'  

df.loc[np.logical_or(df.county=='Tulare & Kern Counties',
                     df.county=='Tulure'),
       'county']='Tulare'  
                     
df.loc[df.county=='Tuolumne and Calaveras counties','county'] = 'Tuolumne' 

df.loc[df.county=='Ventura  Fire Department','county'] = 'Ventura' 

# There are no values for Sierra county, the samples are removed

boolean_drop = df.county=="Sierra"
index_drop = [i for i,x in enumerate(boolean_drop) if x] 
df.drop(index_drop, inplace=True)  
df.reset_index(inplace=True,drop=True)

# There are wrong values that can't be in California

minLatCal = 32.55
maxLatCal = 42.126
minLonCal = -125.05
maxLonCal = -114.02

boolean_drop = np.logical_or(df.latitude<minLatCal,
               np.logical_or(df.latitude>maxLatCal,
               np.logical_or(df.longitude<minLonCal,
                             df.longitude>maxLonCal
                             )))

index_drop = [i for i,x in enumerate(boolean_drop) if x] 
df.drop(index_drop, inplace=True)  
df.reset_index(inplace=True,drop=True)

# Obtain maximum and minimum longitude and latitude for each county

counties = list(set(df.county))
counties.sort()

boundaries_table = pd.DataFrame({'county':counties,
                                 'minLat':df.groupby('county')['latitude'].min().values,
                                 'maxLat':df.groupby('county')['latitude'].max().values,
                                 'minLon':df.groupby('county')['longitude'].min().values,
                                 'maxLon':df.groupby('county')['longitude'].max().values
                               })
    
boundaries_table.set_index('county', inplace=True, drop= True)

# Fill missing coordinates with random coordinates within the county limits
    
boolean_fill = df.longitude.isnull()   
index_fill = [i for i,x in enumerate(boolean_fill) if x] 

for i in index_fill:
    
    df['longitude'][i] = np.random.uniform(boundaries_table.minLon[df.county[i]],
                                          boundaries_table.maxLon[df.county[i]])
    df['latitude'][i] = np.random.uniform(boundaries_table.minLat[df.county[i]],
                                          boundaries_table.maxLat[df.county[i]])

# Get weather data
# Set dates

df.startDate = pd.to_datetime(df.startDate)
df.finishDate = pd.to_datetime(df.finishDate)

url_base = 'http://aviationweather.gov/adds/dataserver_current/httpparam?dataSource=stations&requestType=retrieve&format=xml&radialDistance=40;'
url_base_2 = 'https://www.wunderground.com/history/airport/'

df['meanTemp']= np.nan
df['maxTemp']= np.nan
df['minTemp']= np.nan
df['dewPoint']= np.nan
df['avgHumidity']= np.nan
df['maxHumidity']= np.nan
df['minHumidity']= np.nan
df['maxWind']= np.nan

for k in range(len(df)):
    
    print('Progress: '+str(k)+'/'+str(len(df)))
    
    lat = df.latitude[k]
    lon = df.longitude[k]
    
    url = url_base+str(lon)+','+str(lat)
    
    source = urllib.request.urlopen(url).read()
    soup = bs4.BeautifulSoup(source,"lxml")
    airports = [x.text for x in soup.find_all('station_id')]
    
    year = df.startDate[k].year
    month = df.startDate[k].month
    day = df.startDate[k].day
    
    for j in airports:
        
        try:
            
            url2 = url_base_2+j+'/'+str(year)+'/'+str(month)+'/'+str(day)+'/DailyHistory.html'
            table = pd.read_html(url2)[0].iloc[:,range(2)]
            table.columns= ['a','b']
            table.set_index('a',drop=True,inplace=True)
            
            if(np.logical_or(table.b['Max Temperature']!=table.b['Max Temperature'],table.b['Max Temperature']=='-')):
                
                continue
            
            else:
                
                df['meanTemp'][k]= table['b']['Mean Temperature']
                df['maxTemp'][k]= table['b']['Max Temperature']
                df['minTemp'][k]= table['b']['Min Temperature']
                df['dewPoint'][k]= table['b']['Dew Point']
                df['avgHumidity'][k]= table['b']['Average Humidity']
                df['maxHumidity'][k]= table['b']['Maximum Humidity']
                df['minHumidity'][k]= table['b']['Minimum Humidity']
                df['maxWind'][k]= table['b']['Max Wind Speed']
                
                break
            
        except:
            
            continue

# Clean weather data
# Clean temperature

temp_keys = ['meanTemp','maxTemp','minTemp','dewPoint']

for j in range(len(df)):
    
    for k in temp_keys:
        
        if(df[k][j]=='-'):
            df[k][j]=np.nan
            
        if(df[k][j]==df[k][j]):
            df[k][j]= int(re.sub(r'°F',"",df[k][j]).strip())

df[temp_keys] = df[temp_keys].astype('float') 

# Clean wind

for j in range(len(df)):
        
    if(df['maxWind'][j]=='-'):
        df['maxWind'][j]=np.nan
        
    if(df['maxWind'][j]==df['maxWind'][j]):
        df['maxWind'][j]= int(re.sub(r'mph',"",df['maxWind'][j]).strip())       

df['maxWind'] = df['maxWind'].astype('float')

# Clean burnt acres data

# Get rid of % burnt and the commas

acres = ['burned_containment','estimated_containment:']

for j in range(len(df)):
    
    for k in acres:
        
        if(df[k][j]==df[k][j]):
            df[k][j]= re.sub(r'\d+ ?%',"",df[k][j])
            df[k][j]= re.sub(r',',"",df[k][j])
            
            try:
                df[k][j]=re.findall(r'\d+',df[k][j])[0]
                
            except:
                df[k][j]=np.nan
                
df['burned_containment'] = df['burned_containment'].astype('float')
df['estimated_containment:'] = df['estimated_containment:'].astype('float')

# Build burnt acres column
                
df['burnt_acres']= np.nan

for j in range(len(df)):
    
    if(df['burned_containment'][j]==df['burned_containment'][j]):
        df['burnt_acres'][j] = df['burned_containment'][j]
        continue
    
    if(df['estimated_containment:'][j]==df['estimated_containment:'][j]):
        df['burnt_acres'][j] = df['estimated_containment:'][j]
        continue
        
del df['burned_containment']
del df['estimated_containment:']      

# Clean the cost column

for i in range(len(df)):
    
    try:
        df['cost'][i] = re.search(r'(\d+,)?(\d+,)?\d+(\.\d+)?',df['cost'][i]).group()
        
        if(len(re.findall(r',',df['cost'][i]))>0):
            df['cost'][i]=float(re.sub(r',','',df['cost'][i]))/1000000
            
        else:
            df['cost'][i] = float(df['cost'][i])
        
    except:
        continue
    
df.cost = df.cost.astype('float')

# Clean injuries

for i in range(len(df)):
    
    try:
        temp = re.findall(r'\d+',df['injuries'][i])
        df['injuries'][i] = sum([int(j) for j in temp])
        
    except:
        continue
    
df.injuries=df.injuries.astype('float')

# Clean destroyed structures

for i in range(len(df)):
    
    try:
        temp = re.findall(r'\d+,?\d*',df['destroyedStructures:'][i])
        temp2 = [re.sub(r',','',k) for k in temp]
        df['destroyedStructures:'][i] = sum([int(j) for j in temp2])
        
    except:
        continue
    
df['destroyedStructures:']=df['destroyedStructures:'].astype('float')
df.rename(columns={'destroyedStructures:':'destroyedStructures'}, inplace=True)

# Threatened structures

for i in range(len(df)):
    
    try:
        temp = re.findall(r'\d+',df['threatenedStructures'][i])
        df['threatenedStructures'][i] = sum([int(j) for j in temp])
        
    except:
        continue
    
df.threatenedStructures=df.threatenedStructures.astype('float')

# Clean resources used

resources = ['airtankers', 'dozers','fireEngines', 'firePersonnel', 
             'fireCrews', 'helicopters', 'waterTenders']

for j in resources:
    
    for i in range(len(df)):
        
        try:
            df[j][i] = re.sub(',','',re.search(r'\d+,?\d*',df[j][i]).group())
            
        except:
            continue

    df[j] = df[j].astype('float')
    
# Mark as fire
    
df['fire'] = True

# Select relevant columns and order
    
df = df[['fire_name','county','cost','burnt_acres','startDate','finishDate',
         'latitude','longitude','injuries','destroyedStructures', 
         'threatenedStructures','airtankers','dozers','fireEngines',
         'firePersonnel','fireCrews','helicopters','waterTenders','meanTemp',
         'maxTemp','minTemp','dewPoint','avgHumidity','maxHumidity',
         'minHumidity','maxWind','fire']]

df.to_csv('fires_final.csv', index = False)

