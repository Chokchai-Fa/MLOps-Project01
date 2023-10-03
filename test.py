import pandas as pd

from joblib import load
import json
import os

from sklearn.metrics import accuracy_score


# Set path for the input (model)
MODEL_DIR =  "./model"
model_file = 'logit_model.joblib'
model_path = os.path.join(MODEL_DIR, model_file)

# Set path for the input (test data)
PROCESSED_DATA_DIR = "./processed_data"
test_data_file = 'test.csv'
test_data_path = os.path.join('PROCESSED_DATA_DIR', test_data_file)



# Load model
logit_model = load(model_path)

# Load data
df = pd.read_csv(test_data_path, sep=",")


# Split data into dependent and independent variables
# Drop useless variables
X_test = df.drop(['target'], axis='columns')
y_test = df['target']

# Predict
logit_predictions = logit_model.predict(X_test)

# Compute test accuracy
test_logit = accuracy_score(y_test,logit_predictions)

# Test accuracy to JSON
test_metadata = {
    'test_acc': test_logit
}


# Set output path
RESULTS_DIR = "./results"
test_results_file = 'test_metadata.json'
results_path = os.path.join(RESULTS_DIR, test_results_file)

# Serialize and save metadata
with open(results_path, 'w') as outfile:
    json.dump(test_metadata, outfile)





