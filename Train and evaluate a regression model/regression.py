#Train and evaluate a regression model
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


# load the training dataset

url = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/daily-bike-share.csv"

bike_data = requests.get(url)

if bike_data.status_code == 200:
    with open("grades.csv", "wb") as f:
        f.write(bike_data.content)
    print("File downloaded successfully.")
else:
    print(f"Failed to download the file. Status code: {bike_data.status_code}")

bike_data = pd.read_csv('daily-bike-share.csv', delimiter=',', header='infer')

#bike_data = pd.read_csv('daily-bike-share.csv')
bike_data.head()

# add a new column named day to the dataframe by extracting the day component from the existing dteday column. 
# The new column represents the day of the month, from 1 to 31.
bike_data['day'] = pd.DatetimeIndex(bike_data['dteday']).day
bike_data.head(32)
#print(bike_data)

#let's start our analysis of the data by examining a few key descriptive statistics. 
# We can use the dataframe's describe method to generate these for the numeric features as well as the rentals label column.

numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
bike_data[numeric_features + ['rentals']].describe()

#visualize the data to understand the distribution

# Get the label column
label = bike_data['rentals']


# Create a figure for 2 subplots (2 rows, 1 column)
fig, ax = plt.subplots(2, 1, figsize = (9,12))

# Plot the histogram   
ax[0].hist(label, bins=100)
ax[0].set_ylabel('Frequency')

# Add lines for the mean, median, and mode
ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)

# Plot the boxplot   
ax[1].boxplot(label, vert=False)
ax[1].set_xlabel('Rentals')

# Add a title to the Figure
fig.suptitle('Rental Distribution')

# Show the figure
#plt.show()

#The plots show that the number of daily rentals ranges from 0 to just over 3,400. 
# However, the mean (and median) number of daily rentals is closer to the low end of that range, 
# with most of the data between 0 and around 2,200 rentals. The few values above this are shown in the box plot as small circles,
# indicating that they are outliers; in other words, unusually high or low values beyond the typical range of most of the data.
#We can do the same kind of visual exploration of the numeric features. Let's create a histogram for each of these.


# Plot a histogram for each numeric feature
for col in numeric_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = bike_data[col]
    feature.hist(bins=100, ax = ax)
    ax.axvline(feature.mean(), color='magenta', linestyle='dashed', linewidth=2)
    ax.axvline(feature.median(), color='cyan', linestyle='dashed', linewidth=2)
    ax.set_title(col)

# plot a bar plot for each categorical feature count
categorical_features = ['season','mnth','holiday','weekday','workingday','weathersit', 'day']

for col in categorical_features:
    counts = bike_data[col].value_counts().sort_index()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    counts.plot.bar(ax = ax, color='steelblue')
    ax.set_title(col + ' counts')
    ax.set_xlabel(col) 
    ax.set_ylabel("Frequency")


#Now that we've explored the data, it's time to use it to train a regression model that uses the 
# features we've identified as potentially predictive to predict the rentals label. 
# The first thing we need to do is to separate the features we want to use to train the model from the label we want it to predict.

# Separate features and labels
X, y = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values, bike_data['rentals'].values
print('Features:',X[:10], '\nLabels:', y[:10], sep='\n')

'''
After separating the dataset, we now have numpy arrays named X containing the features and y containing the labels.

We could train a model using all of the data, but it's common practice in supervised learning to split the data into two subsets: a (typically larger) set with which to train the model, and a smaller "hold-back" set with which to validate the trained model. This allows us to evaluate how well the model performs when used with the validation dataset by comparing the predicted labels to the known labels. It's important to split the data randomly (rather than say, taking the first 70% of the data for training and keeping the rest for validation). This helps ensure that the two subsets of data are statistically comparable (so we validate the model with data that has a similar statistical distribution to the data on which it was trained).

To randomly split the data, we'll use the train_test_split function in the scikit-learn library. This library is one of the most widely used machine-learning packages for Python.
'''

from sklearn.model_selection import train_test_split

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

print ('Training Set: %d rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))

'''
Now we have the following four datasets:

X_train: The feature values we'll use to train the model
y_train: The corresponding labels we'll use to train the model
X_test: The feature values we'll use to validate the model
y_test: The corresponding labels we'll use to validate the model
Now we're ready to train a model by fitting a suitable regression algorithm to the training data. We'll use a linear regression algorithm, a common starting point for regression that works by trying to find a linear relationship between the X values and the y label. The resulting model is a function that conceptually defines a line where every possible X and y value combination intersect.

In Scikit-Learn, training algorithms are encapsulated in estimators, and in this case, we'll use the LinearRegression estimator to train a linear regression model.
'''
# Fit a linear regression model on the training set
model = LinearRegression().fit(X_train, y_train)
print (model)

'''
# Fit a linear regression model on the training set
model = LinearRegression().fit(X_train, y_train)
print (model)
'''

predictions = model.predict(X_test)
np.set_printoptions(suppress=True)
print('Predicted labels: ', np.round(predictions)[:10])
print('Actual labels   : ' ,y_test[:10])

'''
Comparing each prediction with its corresponding "ground truth" actual value isn't a very efficient
way to determine how well the model is predicting. Let's see if we can get a better indication by visualizing
a scatter plot that compares the predictions to the actual labels. We'll also overlay a trend line to get a 
general sense for how well the predicted labels align with the true labels.
'''
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Daily Bike Share Predictions')
# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')

#plt.show()
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)

rmse = np.sqrt(mse)
print("RMSE:", rmse)

r2 = r2_score(y_test, predictions)
print("R2:", r2)
