import json
import os
import requests


def download_file(url, output_dir, output_filename):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"File' {output_filename}' already exists in '{output_dir}'. Skipping download.")
    else:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.text
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(data)
            print(f"File '{output_filename}' downloaded successfully to '{output_dir}'.")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")



url ="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/1r_mM6ZPYNxcFv65QkzubA/California-Culinary-Map.txt"
download_file(url, output_dir = "../data/raw", output_filename = "California-Culinary-Map.txt")





