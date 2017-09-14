import numpy as np 
import pandas as pd 


# read data
category = pd.read_csv('category_tree.csv')
events = pd.read_csv('events.csv')
item1 = pd.read_csv('item_properties_part1.csv')
item2 = pd.read_csv('item_properties_part2.csv')
total_item = pd.concat([item1, item2])

# observing how each dataset looks like in order to proceed next data cleaning
print(category.head(n=10))
print(events.head(n=10))
print(total_item.head(n=10))

# finding out how many rows there are 
print(len(category))
print(len(events))
print(len(total_item))

# in this case, by observing a few lines of each dataset, 
# we are trying to predict 'event' in events csv, especially 'transaction', meaning
# the customer bought a certain product. Thru a machine learning model,
# we are going to predict whether it can predict an unknown customer's behavior 
# whether he/she will buy a certain product 

# drop NAs
category = category.dropna()
total_item = total_item.dropna()

# I should not be dropping NaN in events because transactionid is NaN unless a product is bought
# need to be careful here!

# 5 point-summary for each dataset
print(category.describe())
print(events.describe())
print(total_item.describe())

# descriptive summary for 'event' in event.csv
# bargraph representing the distribution of 'event'
import matplotlib.pyplot as plt


event_counts = events['event'].value_counts()
fig, ax = plt.subplots()

event_unique = ['view', 'addtocart', 'transaction']
index = np.arange(len(event_unique))

bar_width = 0.35
opacity = 1
error_config = {'ecolor': '0.3'}

rects = plt.bar(index, [event_counts['view'], event_counts['addtocart'], event_counts['transaction']],
                    bar_width,
                    alpha=opacity,
                    color='b',
                    error_kw=error_config,
                    label='Event')
                 
plt.xticks(index, event_unique)
plt.legend()
plt.tight_layout()
plt.show()


# data cleaning 
# convert categorical data to numerical values
# droping transactionid because transactionid exists when a customer bought a product
# having this column in the model will not be a good determinant in having a good predictive performance
from sklearn.preprocessing import LabelEncoder


events = events[['timestamp','visitorid','itemid','event']]
labelencoder_X=LabelEncoder()
events['event'] = labelencoder_X.fit_transform(events['event'])
print(events.head())

# Running a Naive Bayes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# Cross-validation 
# 80% of data devoted to training, 20% of data devoted to testing
X_train, X_test, y_train, y_test = train_test_split(events[['timestamp','visitorid', 'itemid']], 
                                                    events['event'],
                                                    test_size = 0.2, random_state = 0)

# Fitting a Naive Bayes Model
gnb = GaussianNB()
fit = gnb.fit(X_train, y_train)

# Testing the naive bayes model's performance 
print(fit.score(X_test, y_test))

# Predicting using the Naive Bayes
print(fit.predict([1433211518578, 911093, 251130]))

rf = RandomForestClassifier()
rf_fit = rf.fit(X_train, y_train)

# Testing the random forest model's performance 
print(rf_fit.score(X_test, y_test))

# Predicting using the Random Forest
print(rf_fit.predict([1433211518578, 911093, 251130]))




# A summary of the model you chose, why you chose it, how you manipulated the data to use it, and 
# of course how it performed.


# I chose the two machine learning models for this dataset. Before training and testing,
# I have looked at the data set for several hours to find out what attributes 
# makes sense for a machine learning model to process. I have tried inner method using
# pd.merge() on the column 'itemid' but the machine learning models were extremely slow
# since the dataset became extremely large, up to 20 millions. My conclusion was that 
# it would only make sense to use only event.csv because it contains a few columns that 
# can be proccessed in machine learning models. 
#
# I used 80-20 cross-validation to train and test for the following two machine learning models.
# First, I used Naive Bayes, which is known to be one of the fastest machine learning algorithms out there and
# uses probability to classify. The performance of the model is .966684868682, which is 
# extremly good but the predictive ability is not reliable because the probability of users 'view'ing 
# is much higher than the probabilities of the other two, 'addtocart', and 'transaction'. 
# The second machine learning model I used is Random Forest model, which is known to be
# one of the most accurate machine learning models out there that do not overfit. I used the 
# same cross-validation set for training and testing for the random forest. The performance
# is slightly worse than the performance of naive bayes but the predictive ability is 
# more reliable than the naive bayes. For the same data [1433211518578, 911093, 251130],
# the naive bayes predicted 2, which is 'view' but the random forest predicted 1, 'transaction',
# which is correct. The random forest model can be possibly used to predict.


# Put what you think is important in it. 


# I have led across many data science and software engineering projects. The one of the most
# important things, especially in data science projects, is data. If data does not make sense
# and is not organized in a clean, meaningful manner, regardless how perfect a machine learning
# model is, it would not make any sense of the data and cannot interpret and deliver a result
# to audience. I think data science is all about data and how clean and meaningful data is.
# I would like to explore how zappos uses data to make a better experience for their customers
# and products. 
#
# Thank you so much for reading,
# Michael Chon



