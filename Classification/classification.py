import pandas as pd
import requests
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# load the training dataset

url = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/diabetes.csv"

diabetes = requests.get(url)

if diabetes.status_code == 200:
    with open("diabetes.csv", "wb") as f:
        f.write(diabetes.content)
    print("File downloaded successfully.")
else:
    print(f"Failed to download the file. Status code: {diabetes.status_code}")

bike_data = pd.read_csv('diabetes.csv', delimiter=',', header='infer')

diabetes = pd.read_csv('diabetes.csv')
diabetes.head()
print(diabetes)

'''This data consists of diagnostic information about some patients who have been tested for diabetes. 
Scroll to the right if necessary, and note that the final column in the dataset (Diabetic) contains the value 0 
for patients who tested negative for diabetes, and 1 for patients who tested positive. 
This is the label that we will train our model to predict; most of the other columns 
(Pregnancies, PlasmaGlucose, DiastolicBloodPressure, and so on) are the features we will use to predict the Diabetic label.

Let's separate the features from the labels - we'll call the features X and the label y:'''

# Separate features and labels
features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
label = 'Diabetic'
X, y = diabetes[features].values, diabetes[label].values

for n in range(0,4):
    print("Patient", str(n+1), "\n  Features:",list(X[n]), "\n  Label:", y[n])
    
# Now let's compare the feature distributions for each label value.


features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
for col in features:
    diabetes.boxplot(column=col, by='Diabetic', figsize=(6,6))
    plt.title(col)
#plt.show()
'''he scikit-learn package contains a large number of functions we can use to build a machine learning model - 
including a train_test_split function that ensures we get a statistically random split of training and test data. 
We'll use that to split the data into 70% for training and hold back 30% for testing.'''

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

print ('Training cases: %d\nTest cases: %d' % (X_train.shape[0], X_test.shape[0]))

'''
Train and Evaluate a Binary Classification Model
We're now ready to train our model by fitting the training features (X_train) to the training labels (y_train). T
here are various algorithms 
we can use to train the model. In this example, we'll use Logistic Regression, which (despite its name) 
is a well-established algorithm for classification. In addition to the training features and labels, 
we'll need to set a regularization parameter. This is used to counteract any bias in the sample, and help the 
model generalize well by avoiding overfitting the model to the training data.

Note: Parameters for machine learning algorithms are generally referred to as hyperparameters.
To a data scientist, parameters are values in the data itself - hyperparameters are defined externally from the data.
'''

# Train the model

# Set regularization rate
reg = 0.01

# train a logistic regression model on the training set
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)
print (model)

'''
from sklearn.linear_model import LogisticRegression
'''

predictions = model.predict(X_test)
print('Predicted labels: ', predictions)
print('Actual labels:    ', y_test)

from sklearn.metrics import accuracy_score

print('Accuracy: ', accuracy_score(y_test, predictions))