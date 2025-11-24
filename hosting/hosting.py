from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Define the Hugging Face Space details for the Streamlit app
hf_space_repo_id = "deepakpathania/tourism-predict-app" # Suggesting a name
repo_type = "space"

# Step 1: Check if the space exists, and create if it doesn't
try:
    api.repo_info(repo_id=hf_space_repo_id, repo_type=repo_type)
    print(f"Space '{hf_space_repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{hf_space_repo_id}' not found. Creating new space...")
    create_repo(repo_id=hf_space_repo_id, repo_type=repo_type, space_sdk="docker", private=False)
    print(f"Space '{hf_space_repo_id}' created.")

# Upload the deployment folder content to the Hugging Face Space
api.upload_folder(
    folder_path="deployment/", # The local folder containing your app files
    repo_id=hf_space_repo_id,         # The target Hugging Face Space ID
    repo_type=repo_type,                      # dataset, model, or space
    path_in_repo="",                          # Upload directly to the root of the space
)
print(f"Deployment files uploaded to Hugging Face Space: {hf_space_repo_id}")
