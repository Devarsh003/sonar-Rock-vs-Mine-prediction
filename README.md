# sonar-Rock-vs-Mine-prediction
# Overview
This project focuses on classifying data into two categories: Rock and Mine. We use machine learning models to train on a dataset and then predict whether new data points are classified as Rock or Mine.
Installation
To get started with this project, clone the repository and install the required dependencies.


git clone https://github.com/Devarsh003/rock-vs-mine-classification.git 
cd rock-vs-mine-classification 
pip install -r requirements.txt

Usage
Training the Model
To train the machine learning model on the dataset, ensure your dataset file (sonar data.csv) is in the root directory of the project. Then, run the following script:

python
Copy code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv('sonar data.csv', header=None)
print(df.head())

# Display the shape of the dataset
print(df.shape)

# Count the values for the target column
print(df[60].value_counts())

# Separate features and target
x = df.drop(columns=60)
y = df[60]

# Display the first few rows of features and target
print(x.head())
print(y.head())

# Label encoding for target variable
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model = model.fit(x_train, y_train)

# Print the accuracy of the model
print(f"Model Accuracy: {model.score(x_test, y_test)}")

# Predict the target for test data
y_prediction = model.predict(x_test)

# Create a confusion matrix
cm_df = confusion_matrix(y_test, y_prediction)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()
Predicting New Data
To predict whether new data points are classified as Rock or Mine, you can use the system function defined in the script:

python
Copy code
# Define a function for prediction on new data
def system(data):
    # Convert input data to a numpy array and reshape for prediction
    x = np.asarray(data)
    y = x.reshape(1, -1)
    prediction = model.predict(y)
    print(prediction)
    if prediction[0] == 1:  # Assuming 'R' is encoded as 1 and 'M' as 0
        print("This is Rock Data")
    else:
        print("This is Mine Data")

# Example data for prediction
data4 = (0.0473, 0.0509, 0.0819, 0.1252, 0.1783, 0.3070, 0.3008, 0.2362, 0.3830, 0.3759,
         0.3021, 0.2909, 0.2301, 0.1411, 0.1582, 0.2430, 0.4474, 0.5964, 0.6744, 0.7969,
         0.8319, 0.7813, 0.8626, 0.7369, 0.4122, 0.2596, 0.3392, 0.3788, 0.4488, 0.6281,
         0.7449, 0.7328, 0.7704, 0.7870, 0.6048, 0.5860, 0.6385, 0.7279, 0.6286, 0.5316,
         0.4069, 0.1791, 0.1625, 0.2527, 0.1903, 0.1643, 0.0604, 0.0209, 0.0436, 0.0175,
         0.0107, 0.0193, 0.0118, 0.0064, 0.0042, 0.0054, 0.0049, 0.0082, 0.0028, 0.0027)

# Predict the class for the example data
system(data4)
Dataset
The dataset used in this project consists of sonar signals represented by various numerical features. The last column contains the labels, where 'R' stands for Rock and 'M' stands for Mine.

Ensure the dataset file (sonar data.csv) is in the root directory of the project.

Model Training
The model training process includes:

Loading the Data: Reading the CSV file containing the dataset.
Preprocessing: Encoding the labels and splitting the data into training and testing sets.
Training the Model: Using Logistic Regression to train the model.
Evaluating the Model: Printing the model's accuracy and visualizing the confusion matrix.
Prediction
The prediction process involves using the trained model to classify new data points. You can input new data to the system function to get predictions.

# Contributing
Contributions are welcome! If you have any ideas, suggestions, or improvements, feel free to open an issue or submit a pull request.

# License
This project is licensed under the MIT License. See the LICENSE file for more details.

