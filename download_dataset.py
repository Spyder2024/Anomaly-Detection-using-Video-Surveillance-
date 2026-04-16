import os
import sys
import subprocess

def install_gdown():
    """Install gdown if it's not already installed."""
    try:
        import gdown
    except ImportError:
        print("gdown not found. Installing gdown...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            print("Successfully installed gdown.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install gdown automatically. Please run 'pip install gdown' manually. Error: {e}")
            sys.exit(1)

def download_dataset():
    """Download the dataset from Google Drive."""
    install_gdown()
    import gdown
    
    # ---------------------------------------------------------
    # PLACE YOUR GOOGLE DRIVE FOLDER OR FILE LINK HERE
    # Note: The link must be shared so that "Anyone with the link can view"
    # ---------------------------------------------------------
    drive_url = "https://drive.google.com/drive/folders/1dOOAwXvIq2r54GnrS7oXYH_gkCu5C74k?usp=sharing" 
    
    if not drive_url.strip():
        print("Error: Please open 'download_dataset.py' and update the 'drive_url' variable with your actual Google Drive link.")
        return

    output_path = "DATASET"
    
    print(f"Downloading dataset to '{output_path}'...")
    try:
        # Check if the url is a link to a folder
        if "drive.google.com/drive/folders/" in drive_url or "folder" in drive_url:
            gdown.download_folder(url=drive_url, output=output_path, quiet=False, use_cookies=False)
            print(f"Folder download completed successfully into '{output_path}'.")
        else:
            # Assuming it's a single zipped file or a specific file URL
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            # gdown will try to guess the filename, but we download it inside the DATASET folder
            output_file = os.path.join(output_path, "dataset.zip")
            gdown.download(url=drive_url, output=output_file, quiet=False, fuzzy=True)
            print(f"Download completed. File saved at: {output_file}")
            print(f"Next step: Please unzip/extract the contents of {output_file} into the '{output_path}' folder.")
            
    except Exception as e:
        print(f"An error occurred during download: {e}")

if __name__ == "__main__":
    download_dataset()
