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
           'CRSElapsedTime','Flights','Cancelled','Unnamed: 0', 'Distance', 'DistGroup', 'ActualElapsedTime']:
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

features = ['Year','Quarter','Month' ,'DayofMonth','DayOfWeek','DepDelay']


from sklearn.preprocessing import StandardScaler

# Separating out the features
x = df.loc[:, features].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA

pca = PCA(n_components=6)
X_pca = pca.fit(x)
plot_explained_variance(pca)

#interestingly, year and quarter have no explaination of variation and can be removed; this mean the flight delayis not impacted by year and quarter at all

import datetime
import time

def time_converter(t):
    x = time.strptime(t.split(',')[0],'%H:%M')
    return int(datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds())



df["RealDepTime"] = df.apply( lambda row: time_converter(row.DepTime), axis=1)


cleaned_df = df[['Month' ,'airline','DayofMonth','DayOfWeek', 'DepDelayGroup','RealDepTime']]
columns = cleaned_df # Declare the columns names
y = columns

#one hot encode airline
temp_df = pd.get_dummies(cleaned_df.airline,prefix='airline')
cleaned_df = pd.concat((cleaned_df,temp_df),axis=1)

cleaned_df['DepDelayGroup'] = cleaned_df['DepDelayGroup'].replace("Delay < -15 minutes","Early")
cleaned_df['DepDelayGroup'] = cleaned_df['DepDelayGroup'].replace("Delay between -15 and -1 minutes","Early")
cleaned_df['DepDelayGroup'] = cleaned_df['DepDelayGroup'].replace("Delay between 15 to 29 minutes","Delay greater than 15 minutes")
cleaned_df['DepDelayGroup'] = cleaned_df['DepDelayGroup'].replace("Delay between 30 to 44 minutes","Delay greater than 15 minutes")
cleaned_df['DepDelayGroup'] = cleaned_df['DepDelayGroup'].replace("Delay between 45 to 59 minutes","Delay greater than 15 minutes")
cleaned_df['DepDelayGroup'] = cleaned_df['DepDelayGroup'].replace("Delay between 60 to 74 minutes","Delay greater than 15 minutes")
cleaned_df['DepDelayGroup'] = cleaned_df['DepDelayGroup'].replace("Delay between 75 to 89 minutes","Delay greater than 15 minutes")
cleaned_df['DepDelayGroup'] = cleaned_df['DepDelayGroup'].replace("Delay between 90 to 104 minutes","Delay greater than 15 minutes")
cleaned_df['DepDelayGroup'] = cleaned_df['DepDelayGroup'].replace("Delay between 105 to 119 minutes","Delay greater than 15 minutes")
cleaned_df['DepDelayGroup'] = cleaned_df['DepDelayGroup'].replace("Delay between 120 to 134 minutes","Delay greater than 15 minutes")
cleaned_df['DepDelayGroup'] = cleaned_df['DepDelayGroup'].replace("Delay between 135 to 149 minutes","Delay greater than 15 minutes")
cleaned_df['DepDelayGroup'] = cleaned_df['DepDelayGroup'].replace("Delay between 150 to 164 minutes","Delay greater than 15 minutes")
cleaned_df['DepDelayGroup'] = cleaned_df['DepDelayGroup'].replace("Delay between 165 to 179 minutes","Delay greater than 15 minutes")
cleaned_df['DepDelayGroup'] = cleaned_df['DepDelayGroup'].replace("Delay >= 180 minutes","Delay greater than 15 minutes")




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
    norm_features = ['Month','DayofMonth','DayOfWeek','RealDepTime' ]
    cleaned_df[norm_features] = (cleaned_df[norm_features]-cleaned_df[norm_features].mean()) / cleaned_df[norm_features].std()
    X = cleaned_df.values # use everything else to predict!

    ## X and y are now numpy matrices, by calling 'values' on the pandas data frames we
    #    have converted them into simple matrices to use with scikit learn
print(X)
print(y)

#use random over sampling to valance the DivActualElapsedTime


# to use the cross validation object in scikit learn, we need to grab an instance
#    of the object and set it up. This object will be able to split our data into
#    training and testing splits
num_cv_iterations = 1
num_instances = len(y)
cv_object = ShuffleSplit(
                         n_splits=num_cv_iterations,
                         test_size  = 0.2)
print( cv_object.split(X,y))
# the indices are the rows used for training and testing in each iteration

X_trainOrig, X_train, y_trainOrig, y_train, X_test, y_test = ([] for i in range(6))

#check to see if the data is imablanced
from collections import Counter

#data is imbalanced
print(sorted(Counter(y).items()))




for train_indices, test_indices in cv_object.split(X,y):
    # I will create new variables here so that it is more obvious what
    # the code is doing (you can compact this syntax and avoid duplicating memory,
    # but it makes this code less readable)
    X_trainOrig = X[train_indices]
    y_trainOrig = y[train_indices]
    #use random oversamping for the training dataset
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_sample(X_trainOrig, y_trainOrig)
    print(sorted(Counter(y_resampled).items()))
    X_train = X_resampled
    y_train = y_resampled

    X_test = X[test_indices]
    y_test = y[test_indices]

#explaniaiton of training split
# divide into testing and training
# a ration of 80:20 would be great for our dataset because it is neither too small nor too bigself.
# an average dataset like this one would not need more than 20% of data as testing because 362 is already enough to capture most of the variation
#  But also since our data is not computationally expensive the test data does not need to be less than 20% either
print(sorted(Counter(y_train).items()))
from sklearn import metrics as mt

# first we create a reusable logisitic regression object
#   here we can setup the object with different learning parameters and constants
#lr_clf = RegularizedLogisticRegression(eta=0.1,iterations=2000) # get object

# now we can use the cv_object that we setup before to iterate through the
#    different training and testing sets. Each time we will reuse the logisitic regression
#    object, but it gets trained on different data each time we use it.



# BLR
#inherit from base class
# from last time, our logistic regression algorithm is given by (including everything we previously had):
from scipy.special import expit
from numpy.linalg import pinv
class BinaryLogisticRegression:
    def __init__(self, eta, iterations=20, C=0.001, optChoice='steepest', reg_choice = "o"):
        self.eta = eta
        self.iters = iterations
        self.C = C
        self.optChoice = optChoice
        self.reg_choice = reg_choice
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
            gradient = gradient.reshape(self.w_.shape)


            l_choice = self.reg_choice
            if l_choice == "o":
                p = 0
            elif l_choice == "l1":
                gradient[1:] += -np.sin(self.w_[1:]) * self.C
            elif l_choice == "l2":
                gradient[1:] += -2 * self.w_[1:] * self.C
            elif l_choice == "both":
                gradient[1:] += (-np.sin(self.w_[1:]) + (-2 * self.w_[1:])) * self.C

            return gradient
        elif self.optChoice == 'stochastic':
            # stochastic gradient calculation
            idx = int(np.random.rand()*len(y)) # grab random instance
            ydiff = y[idx]-self.predict_proba(X[idx],add_bias=False) # get y difference (now scalar)
            gradient = X[idx] * ydiff[:,np.newaxis] # make ydiff a column vector and multiply through
            gradient = gradient.reshape(self.w_.shape)


            l_choice = self.reg_choice
            if l_choice == "o":
                p = 0
            elif l_choice == "l1":
                gradient[1:] += -np.sin(self.w_[1:]) * self.C
            elif l_choice == "l2":
                gradient[1:] += -2 * self.w_[1:] * self.C
            elif l_choice == "both":
                gradient[1:] += (-np.sin(self.w_[1:]) + (-2 * self.w_[1:])) * self.C


            return gradient
        elif self.optChoice == 'newtonHessian':
            g = self.predict_proba(X,add_bias=False).ravel() # get sigmoid value for all classes
            hessian = X.T @ np.diag(g*(1-g)) @ X - 2 * self.C # calculate the hessian
            ydiff = y-g # get y difference
            gradient = np.sum(X * ydiff[:,np.newaxis], axis=0) # make ydiff a column vector and multiply through
            gradient = gradient.reshape(self.w_.shape)

            l_choice = self.reg_choice
            if l_choice == "o":
                p = 0
            elif l_choice == "l1":
                gradient[1:] += -np.sin(self.w_[1:]) * self.C
            elif l_choice == "l2":
                gradient[1:] += -2 * self.w_[1:] * self.C
            elif l_choice == "both":
                gradient[1:] += (-np.sin(self.w_[1:]) + (-2 * self.w_[1:])) * self.C

            return pinv(hessian) @ gradient

#Logistic Regression
class LogisticRegression:
    def __init__(self, eta, iterations=20, C=0.001, optChoice='steepest', reg_choice="o"):
        self.eta = eta
        self.iters = iterations
        self.C = C
        self.optChoice = optChoice
        self.reg_choice = reg_choice
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
            blr = VectorBinaryLogisticRegression(self.eta,self.iters,self.C,self.optChoice, self.reg_choice)
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

import numpy as np
def getCArray(beginC, endC, stepSize):
    cArr = []
    for i in np.arange(beginC, endC, stepSize).tolist():
        cArr.append(i)
    return cArr
cmArr = []
def find_best_C(beginC, endC, stepSize, X_train, y_train,X_test,y_test, regression):
    accuracyArr = []
    for i in np.arange(beginC, endC, stepSize).tolist():
        #Choose the optimization and the L term
        if (regression == "lr_steep0"):
            lr = LogisticRegression(eta=0.1, C = i)
        elif (regression == "lr_steep1"):
            lr = LogisticRegression(eta=0.1,C = i,reg_choice = "l1")
        elif (regression == "lr_steep2"):
            lr = LogisticRegression(eta=0.1, C = i, reg_choice = "l2")
        elif (regression == "lr_steepb"):
            lr = LogisticRegression(eta=0.1, C= i, reg_choice = "both")
        elif (regression == "lr_scho0"):
            lr = LogisticRegression(eta=0.1,iterations=1500,C = i, optChoice = 'stochastic')
        elif (regression == "lr_scho1"):
            lr = LogisticRegression(eta=0.1,iterations=1500, C = i, optChoice = 'stochastic',reg_choice = "l1")
        elif (regression == "lr_scho2"):
            lr = LogisticRegression(eta=0.1,iterations=1500, C = i, optChoice = 'stochastic',reg_choice = "l2")
        elif (regression == "lr_schob"):
            lr = LogisticRegression(eta=0.1,iterations=1500, C = i, optChoice = 'stochastic',reg_choice = "both")
        elif (regression == "lr_nh0"):
            lr = LogisticRegression(eta=0.1,iterations=1, C = i, optChoice = 'newtonHessian')
        elif (regression == "lr_nh1"):
            lr = LogisticRegression(eta=0.1,iterations=1, C = i, optChoice = 'newtonHessian',reg_choice = "l1")
        elif (regression == "lr_nh2"):
            lr = LogisticRegression(eta=0.1,iterations=1, C = i, optChoice = 'newtonHessian',reg_choice = "l2")
        elif (regression == "lr_nhb"):
            lr = LogisticRegression(eta=0.1,iterations=1, C = i, optChoice = 'newtonHessian',reg_choice = "both")
        lr.fit(X_train,y_train)  # train object
        y_hat = lr.predict(X_test) # get test set precitions
        acc = mt.accuracy_score(y_test,y_hat)
        accuracyArr.append(acc)
        cmArr.append(metrics.confusion_matrix(y_test, y_hat))
    return accuracyArr


regListName = ["Steepest-Orig :", "Steepest-1 :", "Steepest-2 :", "Steepest-B :", "Stochastic-Orig: ", "Stochastic-1: ", "Stochastic-2: ", "Stochastic-B: ", "NewtonHessian-Orig: ", "NewtonHessian-1: ", "NewtonHessian-2: ", "NewtonHessian-B: "]
regList = ["lr_steep0", "lr_steep1", "lr_steep2", "lr_steepb", "lr_scho0", "lr_scho1", "lr_scho2", "lr_schob", "lr_nh0", "lr_nh1", "lr_nh2", "lr_nhb"]
cList = [0.001,1,0.01]
i = 0
bestC = []
bestAccuracyScore = []
bestCM = []
for r in regList:
    regArr = find_best_C(beginC = cList[0], endC = cList[1], stepSize = cList[2], X_train = X_train, y_train = y_train, X_test = X_test,y_test = y_test, regression = r)
    cArr = getCArray(beginC = cList[0], endC = cList[1], stepSize = cList[2])
    print(regListName[i])
    plt.scatter(cArr, regArr)
    plt.xlabel("C Value")
    plt.ylabel("Accuracy Value")
    plt.title("C Value v Accuracy")
    plt.show()
    print("max accuracy: " , max(regArr))
    bestAccuracyScore.append(max(regArr))
    c_value_index = regArr.index(max(regArr))
    bestC.append(cArr[c_value_index])
    bestCM.append(cmArr[c_value_index])
    print("c value: ", cArr[c_value_index])
    cmArr = []
    i+=1


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
def best_c_confusion(i):

    cm = bestCM[i]
    acc = bestAccuracyScore[i]
    plt.figure(figsize=(10,5))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = regListName[i] + 'Accuracy Score: {0}'.format(acc)
    plt.title(all_sample_title, size = 15)
    plt.show()

#show confusion matrix for each method
i = 0
for r in regList:
    best_c_confusion(i)
    i+=1


def plot_confusion_scikit(y,yhat):
    cm = metrics.confusion_matrix(y, yhat)
    plt.figure(figsize=(10,5))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y,yhat))
    plt.title(all_sample_title, size = 15)
    plt.show()

from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.metrics import accuracy_score
lr_sk = SKLogisticRegression() # all params default
lr_sk.fit(X_train,y_train)
yhat = lr_sk.predict(X_test)
print('Accuracy of: ',accuracy_score(y_test,yhat))
#show confusion matrix for scikit learn
plot_confusion_scikit(y_test,yhat)
#show confusion matrix and accuracy score for best regression
i = bestAccuracyScore.index(max(bestAccuracyScore))
best_c_confusion(i)
