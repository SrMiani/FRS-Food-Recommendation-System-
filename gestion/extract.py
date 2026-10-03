import ast
import json
import os

from PIL import Image
import zipfile
import requests

from llm_model import safe_llm_call, vision_model_call
from prompt_builders import image_caption_prompt_builder, review_context_image_caption_prompt_builder


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





"""
file_path="Recipes.json"
with open(file_path, "r", encoding="utf-8") as f:
    recipes_data = json.load(f)

for key,values in recipes_data[0].items():
    print(f"{key}: type{type(values)} : {values}")

"""


### Step 1.3: Show the image of the first recipe (recipe1)
#url_image_recipe = f"synthetic_recipe_images/recipe{recipes_data[0]['id']}.png"

#Image.open(url_image_recipe)



#Generamos la lista aumentada de recetas con imágenes
"""
for i in range(len(recipes_data)):
    recipe_name = recipes_data[i]['name']


    image_caption_system_msg, image_caption_prompt = image_caption_prompt_builder(recipe_name)

    image_path = f"synthetic_recipe_images/recipe{recipes_data[i]['id']}.png"

    image_caption = safe_llm_call(vision_model_call, image_path, image_caption_prompt, image_caption_system_msg)

    recipes_data[i]['image_description'] = image_caption

print ("ALL DONE")

"""

"""
filename = "Recipes_with_image_descriptions.json"
with open(filename,"w",encoding="utf-8") as f:
    json.dump(recipes_data,f,ensure_ascii=False,indent=4)

    


  



for key,value in reviews_data[0].items():
    print(f"{key}: type{type(value)} : {value}")


images_urls = ast.literal_eval(reviews_data[0]['images'])

response = requests.get(images_urls[0])

if response.status_code == 200:
    with open("rewview_image_placeholder.jpg", "wb") as f:
        f.write(response.content)

Image.open("rewview_image_placeholder.jpg")


test_prompt_review = reviews_data[0]['text']
prompt_image_caption_system_msg, prompt_image_caption_prompt = review_context_image_caption_prompt_builder(test_prompt_review)

response = safe_llm_call(vision_model_call, "rewview_image_placeholder.jpg", prompt_image_caption_prompt, prompt_image_caption_system_msg)
print(response)

"""



filename_reviews = "Synthetic-User-Reviews.json"
with open(filename_reviews,"r",encoding="utf-8") as f:
    reviews_data = json.load(f)



for i in range(len(reviews_data)):
     review_images_urls = ast.literal_eval(reviews_data[i]['images'])
     review_image_captions= []
     if len(review_images_urls) > 0:
          for img_url in review_images_urls:
               try:
                    response=requests.get(img_url)
                    if response.status_code == 200:
                         print("Successfully downloaded")
               except Exception as e:
                    print(f"Error downloading image: {e}")
                    continue
               image = response.content
               with open("review_image_placeholder.jpg","wb") as f:
                    f.write(image)

               test_prompt_review = reviews_data[i]['text']
               prompt_image_caption_system_msg, prompt_image_caption_prompt = review_context_image_caption_prompt_builder(test_prompt_review)
               review_image_caption = safe_llm_call(vision_model_call, "review_image_placeholder.jpg", prompt_image_caption_prompt, prompt_image_caption_system_msg)
               review_image_captions.append(review_image_caption)

     reviews_data[i]['image_captions'] = review_image_captions
print ("ALL DONE")



filename = "augmented_Synthetic-User-Reviews_with_image_captions.json"
with open(filename,"w",encoding="utf-8") as f:
     json.dump(reviews_data,f,ensure_ascii=False,indent=4)



     















