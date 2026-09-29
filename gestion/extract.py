import json
import os
import requests


url ="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/1r_mM6ZPYNxcFv65QkzubA/California-Culinary-Map.txt"
output_dir="../data/raw"

os.makedirs(output_dir, exist_ok=True)

output_path=os.path.join(output_dir,"California-Culinary-Map.txt")


response = requests.get(url)



if response.status_code == 200:
    data = response.text
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(data)

    print("File downloaded successfully.")

