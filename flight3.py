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

import datetime
import time

def time_converter(t):
    x = time.strptime(t.split(',')[0],'%H:%M')
    return int(datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds())



df["RealDepTime"] = df.apply( lambda row: time_converter(row.DepTime), axis=1)


cleaned_df = df[['Month' ,'airline','DayofMonth','DayOfWeek', 'ActualElapsedTime', 'DepDelayGroup','RealDepTime']]
columns = cleaned_df # Declare the columns names
y = columns

#one hot encode airline
temp_df = pd.get_dummies(cleaned_df.airline,prefix='airline')
cleaned_df = pd.concat((cleaned_df,temp_df),axis=1)

if 'airline' in cleaned_df:
    del cleaned_df['airline'] # get rid of the original category as it is now one-hot encoded
#cleaned_df.head()
#change dep delay categories to numberical data
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
cleaned_df['DepDelayGroupA'] = number.fit_transform(cleaned_df['DepDelayGroup'].astype('str'))
#cleaned_df.head()

print(cleaned_df['DepDelayGroupA'].nunique())
print(df['DepDelayGroup'].nunique())

#show the value that coorsponds to original groups
comp_df = cleaned_df[['DepDelayGroup' ,'DepDelayGroupA']]
print(comp_df.drop_duplicates())

#drop original category in cleaned_df
if 'DepDelayGroup' in cleaned_df:
    del cleaned_df['DepDelayGroup'] # get rid of the original category as it is now one-hot encoded
cleaned_df.head()
# create training and testing vars
##y_train, y_test, X_train, X_test = train_test_split(cleaned_df, y, test_size=0.2)
#remove label from training dataset
#X_train = X_train.drop(['DepDelayGroup'], axis=1)
#print(X_train.shape)
#print(X_test.shape)
from sklearn.model_selection import ShuffleSplit

# we want to predict the X and y data as follows:
if 'DepDelayGroupA' in cleaned_df:
    y = cleaned_df['DepDelayGroupA'].values # get the labels we want
    del cleaned_df['DepDelayGroupA'] # get rid of the class label
    norm_features = ['Month','DayofMonth','DayOfWeek','ActualElapsedTime','RealDepTime' ]
    cleaned_df[norm_features] = (cleaned_df[norm_features]-cleaned_df[norm_features].mean()) / cleaned_df[norm_features].std()
    X = cleaned_df.values # use everything else to predict!

    ## X and y are now numpy matrices, by calling 'values' on the pandas data frames we
    #    have converted them into simple matrices to use with scikit learn
print(X)
print(y)
# to use the cross validation object in scikit learn, we need to grab an instance
#    of the object and set it up. This object will be able to split our data into
#    training and testing splits
num_cv_iterations = 1
num_instances = len(y)
cv_object = ShuffleSplit(
                         n_splits=num_cv_iterations,
                         test_size  = 0.2)

print(cv_object)

# divide into testing and training
# a ration of 80:20 would be great for our dataset because it is neither too small nor too bigself.
# an average dataset like this one would not need more than 20% of data as testing because 362 is already enough to capture most of the variation
#  But also since our data is not computationally expensive the test data does not need to be less than 20% either

from sklearn import metrics as mt

# first we create a reusable logisitic regression object
#   here we can setup the object with different learning parameters and constants
#lr_clf = RegularizedLogisticRegression(eta=0.1,iterations=2000) # get object

# now we can use the cv_object that we setup before to iterate through the
#    different training and testing sets. Each time we will reuse the logisitic regression
#    object, but it gets trained on different data each time we use it.


#BLR base
import numpy as np
class BinaryLogisticRegressionBase:
    # private:
    def __init__(self, eta, iterations=20, optChoice = 'steepest'):
        self.eta = eta
        self.iters = iterations
        self.optChoice = optChoice
        # internally we will store the weights as self.w_ to keep with sklearn conventions

    def __str__(self):
        return 'Base Binary Logistic Regression Object, Not Trainable'

    # convenience, private and static:
    @staticmethod
    def _sigmoid(theta):
        return 1/(1+np.exp(-theta))

    @staticmethod
    def _add_bias(X):
        return np.hstack((np.ones((X.shape[0],1)),X)) # add bias term

    # public:
    def predict_proba(self,X,add_bias=True):
        # add bias term if requested
        Xb = self._add_bias(X) if add_bias else X
        return self._sigmoid(Xb @ self.w_) # return the probability y=1

    def predict(self,X):
        return (self.predict_proba(X)>0.5) #return the actual prediction
#

# BLR
#inherit from base class
# from last time, our logistic regression algorithm is given by (including everything we previously had):
from scipy.special import expit
from numpy.linalg import pinv
class BinaryLogisticRegression:
    def __init__(self, eta, iterations=20, C=0.001, optChoice='steepest'):
        self.eta = eta
        self.iters = iterations
        self.C = C
        self.optChoice = optChoice
        # internally we will store the weights as self.w_ to keep with sklearn conventions

    def __str__(self):
        if(hasattr(self,'w_')):
            return 'Binary Logistic Regression Object with coefficients:\n'+ str(self.w_) # is we have trained the object
        else:
            return 'Untrained Binary Logistic Regression Object'

    # convenience, private:
    @staticmethod
    def _add_bias(X):
        return np.hstack((np.ones((X.shape[0],1)),X)) # add bias term

    @staticmethod
    def _sigmoid(theta):
        # increase stability, redefine sigmoid operation
        return expit(theta) #1/(1+np.exp(-theta))

    # vectorized gradient calculation with regularization using L2 Norm
    def _get_gradient(self,X,y):
        ydiff = y-self.predict_proba(X,add_bias=False).ravel() # get y difference
        gradient = np.mean(X * ydiff[:,np.newaxis], axis=0) # make ydiff a column vector and multiply through

        gradient = gradient.reshape(self.w_.shape)
        gradient[1:] += -2 * self.w_[1:] * self.C

        return gradient

    # public:
    def predict_proba(self,X,add_bias=True):
        # add bias term if requested
        Xb = self._add_bias(X) if add_bias else X
        return self._sigmoid(Xb @ self.w_) # return the probability y=1

    def predict(self,X):
        return (self.predict_proba(X)>0.5) #return the actual prediction


    def fit(self, X, y):
        Xb = self._add_bias(X) # add bias term
        num_samples, num_features = Xb.shape

        self.w_ = np.zeros((num_features,1)) # init weight vector to zeros

        # for as many as the max iterations
        for _ in range(self.iters):
            gradient = self._get_gradient(Xb,y)
            self.w_ += gradient*self.eta # multiply by learning rate



    # public:
    def fit(self, X, y):
        Xb = self._add_bias(X) # add bias term
        num_samples, num_features = Xb.shape

        self.w_ = np.zeros((num_features,1)) # init weight vector to zeros

        # for as many as the max iterations
        for _ in range(self.iters):
            gradient = self._get_gradient(Xb,y)
            self.w_ += gradient*self.eta # multiply by learning rate

#Vector BinaryLogisticRegression
class VectorBinaryLogisticRegression(BinaryLogisticRegression):
    # inherit from our previous class to get same functionality
    @staticmethod
    def _sigmoid(theta):
        # increase stability, redefine sigmoid operation
        return expit(theta) #1/(1+np.exp(-theta))

    # but overwrite the gradient calculation
    def _get_gradient(self,X,y):
        if self.optChoice == 'steepest':
            ydiff = y-self.predict_proba(X,add_bias=False).ravel() # get y difference
            gradient = np.mean(X * ydiff[:,np.newaxis], axis=0) # make ydiff a column vector and multiply through
            return gradient.reshape(self.w_.shape)
        if self.optChoice == 'stochastic':
            # stochastic gradient calculation
            idx = int(np.random.rand()*len(y)) # grab random instance
            ydiff = y[idx]-self.predict_proba(X[idx],add_bias=False) # get y difference (now scalar)
            gradient = X[idx] * ydiff[:,np.newaxis] # make ydiff a column vector and multiply through
            gradient = gradient.reshape(self.w_.shape)
            gradient[1:] += -2 * self.w_[1:] * self.C
            return gradient
        if self.optChoice == 'newtonHessian':
            g = self.predict_proba(X,add_bias=False).ravel() # get sigmoid value for all classes
            hessian = X.T @ np.diag(g*(1-g)) @ X - 2 * self.C # calculate the hessian
            ydiff = y-g # get y difference
            gradient = np.sum(X * ydiff[:,np.newaxis], axis=0) # make ydiff a column vector and multiply through
            gradient = gradient.reshape(self.w_.shape)
            gradient[1:] += -2 * self.w_[1:] * self.C
            return pinv(hessian) @ gradient

#Logistic Regression
class LogisticRegression:
    def __init__(self, eta, iterations=20, C=0.001, optChoice='steepest'):
        self.eta = eta
        self.iters = iterations
        self.C = C
        self.optChoice = optChoice
        # internally we will store the weights as self.w_ to keep with sklearn conventions

    def __str__(self):
        if(hasattr(self,'w_')):
            return 'MultiClass Logistic Regression Object with coefficients:\n'+ str(self.w_) # is we have trained the object
        else:
            return 'Untrained MultiClass Logistic Regression Object'

    def fit(self,X,y):
        num_samples, num_features = X.shape
        self.unique_ = np.unique(y) # get each unique class value
        num_unique_classes = len(self.unique_)
        self.classifiers_ = [] # will fill this array with binary classifiers

        for i,yval in enumerate(self.unique_): # for each unique value
            y_binary = y==yval # create a binary problem
            # train the binary classifier for this class
            blr = VectorBinaryLogisticRegression(self.eta,self.iters,self.C,self.optChoice)
            blr.fit(X,y_binary)
            # add the trained classifier to the list
            self.classifiers_.append(blr)

        # save all the weights into one matrix, separate column for each class
        self.w_ = np.hstack([x.w_ for x in self.classifiers_]).T

    def predict_proba(self,X):
        probs = []
        for blr in self.classifiers_:
            probs.append(blr.predict_proba(X)) # get probability for each classifier

        return np.hstack(probs) # make into single matrix

    def predict(self,X):
        return np.argmax(self.predict_proba(X),axis=1) # take argmax along row

#Test
lr_steep = LogisticRegression(eta=0.1) # get object
lr_shco = LogisticRegression(eta=0.1,iterations=1500,optChoice = 'stochastic') # get object
lr_nh = LogisticRegression(eta=0.1,iterations=1, optChoice = 'newtonHessian')
iter_num=0
# the indices are the rows used for training and testing in each iteration
for train_indices, test_indices in cv_object.split(X,y):
    # I will create new variables here so that it is more obvious what
    # the code is doing (you can compact this syntax and avoid duplicating memory,
    # but it makes this code less readable)
    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    lr_steep.fit(X_train,y_train)  # train object
    y_hat = lr_steep.predict(X_test) # get test set precitions
    acc = mt.accuracy_score(y_test,y_hat)
    conf = mt.confusion_matrix(y_test,y_hat)
    print("====Steepest====")
    print("accuracy", acc )
    print("confusion matrix\n",conf)

    lr_shco.fit(X_train,y_train)  # train object
    y_hat = lr_shco.predict(X_test) # get test set precitions
    acc = mt.accuracy_score(y_test,y_hat)
    conf = mt.confusion_matrix(y_test,y_hat)
    print("====Stochastic====")
    print("accuracy", acc )
    print("confusion matrix\n",conf)

    lr_nh.fit(X_train,y_train)  # train object
    y_hat = lr_nh.predict(X_test) # get test set precitions
    acc = mt.accuracy_score(y_test,y_hat)
    conf = mt.confusion_matrix(y_test,y_hat)
    print("====NewtonHessian====")
    print("accuracy", acc )
    print("confusion matrix\n",conf)


from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.metrics import accuracy_score
lr_sk = SKLogisticRegression() # all params default

lr_sk.fit(X,y)
yhat = lr_sk.predict(X)
print('Accuracy of: ',accuracy_score(y,yhat))
