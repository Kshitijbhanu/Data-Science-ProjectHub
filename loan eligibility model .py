# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv(r"C:\Users\Meghna\Documents\Portfolio\loan eligibility\loan-train.csv")

# Data preprocessing
# Drop irrelevant columns or features
data.drop('Loan_ID', axis=1, inplace=True)

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data)

# Split the data into features (X) and target variable (y)
X = data.drop('Loan_Status_Y', axis=1)
y = data['Loan_Status_Y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict loan eligibility on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model:", accuracy)


