import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore', DeprecationWarning)
%matplotlib inline

import missingno as mn

import pandas as pd
import numpy as np

df = pd.read_csv('DallasToChicago.csv') # read in the csv file
print('Pandas:', pd.__version__)
print('Numpy:',np.__version__)

#Remove attributes that are not useful for us
for col in ['TailNum','FlightNum','OriginAirportID',
           'OriginCityName','OriginState','OriginStateName','DestAirportID','DestCityName','DestState','DestStateName','CRSDepTime',
           'DepDelayMinutes','TaxiIn','TaxiOut','CRSArrTime','ArrTime','ArrDelay','ArrDelayMinutes','ArrDelayGroup','ATimeBlk','CancellationReason',
            'Diverted', 'AirTime','CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay','FirstDepTime1','FirstDepTime2',
            'FirstDepTime','TotalAddGTime','LongestAddGTime','DivAirportLandings','DivReachedDest','DivActualElapsedTime','DivArrDelay','DivDistance',
           'CRSElapsedTime','Flights','Cancelled','Unnamed: 0', 'Distance', 'DistGroup']:
    if col in df:
        del df[col]

print(df.dtypes)

''' Explanation

The data we are focusing on is mainly flight delay time data so we chose to keep the colums of:
* dayname (the day of the week of the flight) -----------------------> ordinal
* airline (the airline of the flight)--------------------------------> nominal
* Origin (origin airport code of the flight)-------------------------> nominal
* Dest (destination airport code of the flight)----------------------> nominal
* DepTime (departure time of the flight)-----------------------------> ratio
* DepDelay (delay of the flight departure in minutes)----------------> interval
* DepDelayGroup (delay of the flight departure grouped by minutes)---> ordinal
* DTimeBlk (delay of the flight grouped by hours)--------------------> ordinal
* ActualElapsedTime (flight time in the air)-------------------------> ratio
* Distance (distance the flight travelled)---------------------------> interval
* DistGroup (distance the flight travelled grouped by miles----------> ordinal
'''
print(df.info(verbose=True, null_counts=True))

#Figure out what data is missing
df.isnull().sum()

mn.matrix(df)

#Remove the missing data
df.dropna(inplace=True)
df.count()
print(df.shape)
df.dtypes
df.head()



#see explained variance
import numpy as np
def plot_explained_variance(pca):
    import plotly
    from plotly.graph_objs import Bar, Line
    from plotly.graph_objs import Scatter, Layout
    from plotly.graph_objs.scatter import Marker
    from plotly.graph_objs.layout import XAxis, YAxis
    plotly.offline.init_notebook_mode() # run at the start of every notebook

    explained_var = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(explained_var)

    plotly.offline.iplot({
        "data": [Bar(y=explained_var, name='individual explained variance'),
                 Scatter(y=cum_var_exp, name='cumulative explained variance')
            ],
        "layout": Layout(xaxis=XAxis(title='Principal components'), yaxis=YAxis(title='Explained variance ratio'))
    })

features = ['Year','Quarter','Month' ,'DayofMonth','DayOfWeek','DepDelay','ActualElapsedTime']


from sklearn.preprocessing import StandardScaler

# Separating out the features
x = df.loc[:, features].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA

pca = PCA(n_components=7)
X_pca = pca.fit(x)
plot_explained_variance(pca)

#interestingly, year and quarter have no explaination of variation and can be removed; this mean the flight delayis not impacted by year and quarter at all
