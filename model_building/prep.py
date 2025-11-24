# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/deepakpathania/tourism-dataset/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier and 'Unnamed: 0' column if it exists
df.drop(columns=['CustomerID', 'Unnamed: 0'], errors='ignore', inplace=True)

df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
print("Gender column updated: 'Fe Male' entries replaced with 'Female'.")
print("New unique values in Gender column:", df['Gender'].unique())

# Encoding categorical columns
label_encoder = LabelEncoder()
for column in df.select_dtypes(include='object').columns:
    df[column] = label_encoder.fit_transform(df[column])

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# Re-using the repo_id from data_register.py which is for this project
# Make sure the repo_id variable is available in the global scope if data_register.py was run.
# For safety, let's explicitly define it if it's not guaranteed to be global after %run
project_repo_id = "deepakpathania/tourism-dataset"

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=project_repo_id, # Use the correct repo ID for the project
        repo_type="dataset"
        # Removed create_pr=1 to directly commit files, making them immediately accessible
    )
