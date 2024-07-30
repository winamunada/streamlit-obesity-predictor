import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Load the dataset
data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Data preprocessing
data_encoded = pd.get_dummies(data, drop_first=True)

# Identify the correct target column name based on inspection
target_column = 'NObeyesdad'  # Ensure this is the correct target column name

# Encode the target column
label_encoder = LabelEncoder()
data[target_column] = label_encoder.fit_transform(data[target_column])

# Split the data into features and target
X = data_encoded.drop([target_column], axis=1)
y = data[target_column]

# Save the list of feature names and the label encoder
feature_names = X.columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    joblib.dump(feature_names, f)
with open('label_encoder.pkl', 'wb') as f:
    joblib.dump(label_encoder, f)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with scaling and logistic regression
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, 'obesity_model.pkl')
