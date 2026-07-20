import gdown

folder_url = "https://drive.google.com/drive/folders/YOUR_FOLDER_ID"
custom_output_path = "path/to/your/destination_folder"

gdown.download_folder(
    url=folder_url,
    output=custom_output_path,
    quiet=False,
    use_cookies=False
)