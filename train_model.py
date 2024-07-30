import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Load the dataset
data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Print the columns to confirm the exact column names
print("Columns in the dataset:", data.columns)

# Data preprocessing
# Encode categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)

# Print the columns after encoding to confirm the exact column names
print("Columns after encoding:", data_encoded.columns)

# Identify the correct target column name by inspecting the output from the above print statements
target_column = 'NObeyesdad' 

# Split the data into features and target
X = data_encoded.drop(target_column, axis=1)
y = data_encoded[target_column]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with scaling and logistic regression
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, 'obesity_model.pkl')
