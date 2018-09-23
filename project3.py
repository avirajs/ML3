from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler

#thousands option to turn string comma into numbers
df = pd.read_csv('FishData.csv', thousands=',')# read in the csv file

df.head()

print(df.info())

#remove columns that never change
df = df.drop(columns=['FirstClassLow', 'SizeInterval', 'TimeStamp'])

#removes weight class features where fish weight class is zero; 60 were not used
df = df.loc[:, (df != 0).any(axis=0)]

#remove
print(df.info())
print(df.describe())

#types seem okay
print (df.dtypes)
print(df.columns)

print(df.median())
print(df.tmt.max())
print(df.tno.max())

#quarter is an important feature
print(df.groupby(by='Quarter').median())
#biggest fish haul
print(df.iloc[df.tmt.idxmax()])








#create new columns
r = re.compile("[T]\d")
weight_classes = list(filter(r.match, df.columns.tolist()))
print(weight_classes)

#recompute weight ranking into smaller, readable data
df["1"]= df.iloc[:, -10:-1].sum(axis=1)
df["2"]= df.iloc[:, -20:-11].sum(axis=1)
df["3"]= df.iloc[:, -30:-21].sum(axis=1)
df["4"]= df.iloc[:, -40:-31].sum(axis=1)
df["5"]= df.iloc[:, -50:-41].sum(axis=1)
df["6"]= df.iloc[:, -60:-51].sum(axis=1)
df["7"]= df.iloc[:, -70:-61].sum(axis=1)
df["8"]= df.iloc[:, -80:-71].sum(axis=1)
df["9"]= df.iloc[:, -90:-81].sum(axis=1)
df = df.drop(columns=weight_classes)












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

features = ['Year', 'Quarter', 'tno', 'tmt']


# Separating out the features
x = df.loc[:, features].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=4)
X_pca = pca.fit(x)
plot_explained_variance(pca)














features = ['Year', 'Quarter', 'tno', 'tmt']
# Separating out the features
x = df.loc[:, features].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])



import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

principalDf.columns = ['PCA component 1', 'PCA component 2']
principalDf.plot.scatter(x='PCA component 1', y='PCA component 2', marker='o',
        alpha=0.7, # opacity
        title="red: ckd, green: not-ckd" )
plt.show()
#the total catch number and total pound number is the same so the variaion explained only needs three features to explain most variation



# divide into testing and training
# a ration of 80:20 would be great for our dataset because it is neither too small nor too bigself.
# a large dataset like this one would not need more than 20% of data as testing because 3000 is already enough to capture most of the variation
# infact even a few less than 20%, around 15% would be okay. But also since our data is not computationally expensive the test data can be left at a higher level of 20% anyways
num_df = df.select_dtypes(include='number')
num_df.shape[0] * .2
columns = num_df # Declare the columns names
y = columns

# create training and testing vars
y_train, y_test, X_train, X_test = train_test_split(num_df, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# fit a model
lm = linear_model.LinearRegression()
model = lm.fit( y_train, X_train)
predictions = lm.predict(X_test)
## The line / model
plt.scatter(y_test, predictions)
print("Score:", model.score(X_test, y_test))
