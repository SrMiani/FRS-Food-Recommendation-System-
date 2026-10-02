import json
import os

import zipfile
import requests

from llm_model import safe_llm_call, vision_model_call
from prompt_builders import image_caption_prompt_builder


#Text Extracting
# Function to split the data into a list of restaurant paragraphs
def get_data_to_list(data):
        restaurant_list = data.split("\n\n")
        restaurant_list = restaurant_list[1:]
        return restaurant_list


def download_file(url, output_dir, output_filename):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"File '{output_filename}' already exists in '{output_dir}'. Skipping download.")
        return output_path

    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(response.text)
        print(f"File '{output_filename}' downloaded successfully to '{output_dir}'.")
        return output_path
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        return None


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/1r_mM6ZPYNxcFv65QkzubA/California-Culinary-Map.txt"

def load_restaurant_data(url=url, output_dir="../data/raw", output_filename="California-Culinary-Map.txt"):
    output_path = download_file(url, output_dir, output_filename)
    with open(output_path, "r", encoding="utf-8") as file:
        data = file.read()
    return get_data_to_list(data)


#Images Extracting

urls = [
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hpTjb6liKBLVHQK0UgMi5A/Recipes.json",
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fQUs9wQ6aB6ts6fmkD2V2w/Synthetic-User-Reviews.json",
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip",
]





file_path="Recipes.json"
with open(file_path, "r", encoding="utf-8") as f:
    recipes_data = json.load(f)

for key,values in recipes_data[0].items():
    print(f"{key}: type{type(values)} : {values}")




### Step 1.3: Show the image of the first recipe (recipe1)
#url_image_recipe = f"synthetic_recipe_images/recipe{recipes_data[0]['id']}.png"

#Image.open(url_image_recipe)



#Generamos la lista aumentada de recetas con imágenes
for i in range(len(recipes_data)):
    recipe_name = recipes_data[i]['name']


    image_caption_system_msg, image_caption_prompt = image_caption_prompt_builder(recipe_name)

    image_path = f"synthetic_recipe_images/recipe{recipes_data[i]['id']}.png"

    image_caption = safe_llm_call(vision_model_call, image_path, image_caption_prompt, image_caption_system_msg)

    recipes_data[i]['image_description'] = image_caption

print ("ALL DONE")



filename = "Recipes_with_image_descriptions.json"
with open(filename,"w",encoding="utf-8") as f:
    json.dump(recipes_data,f,ensure_ascii=False,indent=4)
















