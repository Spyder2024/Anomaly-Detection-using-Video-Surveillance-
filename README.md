# Anomaly Detection using Video Surveillance

This repository contains the codebase for Anomaly Detection using Video Surveillance.

## Dataset

Due to the massive size of the video dataset (~24.5 GB) and GitHub's strict file size limits, the raw video files are not tracked in this repository. 

Instead, the dataset is hosted externally on Google Drive.

### How to get the Dataset

We have provided an automated script to download the dataset straight into your workspace.

1. First, ensure you have Python installed.
2. Run the provided script to automatically fetch the dataset:

   ```bash
   python download_dataset.py
   ```

The script will automatically download the required files, and place them into the `DATASET/` folder in the root of the project.

*(Note: The script uses the `gdown` Python package under the hood. It will attempt to install it automatically if you do not have it, or you can install it manually by running `pip install gdown`)*
