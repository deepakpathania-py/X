# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Assuming MLflow UI is running locally, if not, update the tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_experiment") # Set the experiment name for the tourism project

api = HfApi(token=os.getenv("HF_TOKEN")) # Ensure HF_TOKEN is set


# Download preprocessed data
Xtrain = pd.read_csv("https://huggingface.co/datasets/deepakpathania/tourism-dataset/resolve/main/Xtrain.csv")
Xtest = pd.read_csv("https://huggingface.co/datasets/deepakpathania/tourism-dataset/resolve/main/Xtest.csv")
ytrain = pd.read_csv("https://huggingface.co/datasets/deepakpathania/tourism-dataset/resolve/main/ytrain.csv").squeeze() # .squeeze() to convert DataFrame to Series
ytest = pd.read_csv("https://huggingface.co/datasets/deepakpathania/tourism-dataset/resolve/main/ytest.csv").squeeze() # .squeeze() to convert DataFrame to Series

# Define features based on tourism.csv
numerical_features = [
    'Age',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'NumberOfTrips',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]
categorical_features = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'ProductPitched',
    'PreferredPropertyStar',
    'MaritalStatus',
    'Passport',
    'OwnCar',
    'PitchSatisfactionScore',
    'Designation'
]

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numerical_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features),
    remainder='passthrough' # Keep other columns if any
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, use_label_encoder=False, eval_metric='logloss')

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='recall') # Use recall for imbalanced target
    grid_search.fit(Xtrain, ytrain)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    print(f"Model saved as artifact at: {model_path}")

    # Log the model with MLflow
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_xgboost_model",
        registered_model_name="XGBoostTourismPredictor", # Register the model
        signature=mlflow.models.signature.infer_signature(Xtest, y_pred_test)
    )
    print("Best model logged to MLflow.")

    # Upload to Hugging Face
    hf_model_repo_id = "deepakpathania/tourism-xgboost-model" # New repo for the model
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=hf_model_repo_id, repo_type=repo_type)
        print(f"Space '{hf_model_repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{hf_model_repo_id}' not found. Creating new space...")
        create_repo(repo_id=hf_model_repo_id, repo_type=repo_type, private=False)
        print(f"Space '{hf_model_repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=f"xgboost_model/{model_path}", # Store inside a folder in the repo
        repo_id=hf_model_repo_id,
        repo_type=repo_type,
        create_pr=1
    )
    print(f"Model uploaded to Hugging Face Hub: {hf_model_repo_id}")
