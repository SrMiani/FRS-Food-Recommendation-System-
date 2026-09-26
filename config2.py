import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import requests
import zipfile
from PIL import Image
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from tenacity import retry, stop_after_attempt, wait_exponential



urls = [
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hpTjb6liKBLVHQK0UgMi5A/Recipes.json",
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fQUs9wQ6aB6ts6fmkD2V2w/Synthetic-User-Reviews.json",
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip",
]

for url in urls:
    filename = url.split("/")[-1]
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"{filename} descargado")

    

with zipfile.ZipFile("synthetic-recipe-images.zip", 'r') as zip_ref:
    zip_ref.extractall()



    ### Your Code Here
### Step 1.1: Load the json file. Define the loaded data as recipe_data.
file_path = "Recipes.json"
with open(file_path, 'r') as file:
    recipe_data = json.load(file)

### Step 1.2: Print each key-value pair of the first recipe. In this format: key (type of value): value
for key, values in recipe_data[0].items():
    print(f"{key}  ({type(values)}) : {values}")

    

### Step 1.3: Show the image of the first recipe (recipe1)
url_image_recipe = f"synthetic_recipe_images/recipe{recipe_data[0]['id']}.png"

Image.open(url_image_recipe)



### Your Code Here: Fill in the blanks
import base64

def vision_llm(system_msg, prompt_txt, image_path):
    #system_msg: input system message for the LLM
    #prompt_txt: input user prompt for the LLM
    #image_path: the file path of the input image

    ### Credentials of the model
    model_id = 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8'
    project_id = "skills-network"
    credentials = Credentials(
                url = "https://us-south.ml.cloud.ibm.com"
                )
    generate_params = {"max_tokens": 300}

    ### Step 2.1: Define the model by ModelInference
    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params=generate_params
    )
    

    ### Step 2.2: Encode the input image to a base64 string
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        
    

    ### Step 2.3: Define the messages for the model
    messages = [
        { "role" : "system" ,"content" : system_msg },
        { 
            "role" : "user", 
            "content" : [
                {"type":"text","text" : prompt_txt},
                {
                    "type":"image_url",
                    "image_url":{"url":f"data:image/png;base64,{encoded_image}"}
                }
            ] 
        }
    ]
    
    
    
    

    ### Step 2.4: Get the response for the messages 

    response = model.chat(messages=messages)
    output_text = response["choices"][0]["message"]["content"]
    

    return output_text



    ### Your Code Here

### Define the food image caption prompts given a food name.
### The food name, as you have noticed, comes from the corresponding recipe data.
### You want to include the food name to ensure the model focuses on it while giving captions.
def image_caption_prompt_template(food_name):
    # food_name: the food name of the recipe

    ### Step 3.1: Design the prompts
    image_caption_system_msg = "Eres un asistente especializado en describir imágenes de comida con detalle, para uso en un sistema de búsqueda de recetas."
    image_caption_prompt_txt = f"""Describe esta imagen de {food_name} centrándote en su apariencia visual: color, textura, presentación y guarniciones visibles.
    Sé conciso: 2-3 frases como máximo, centrándote en detalles que ayuden a identificar visualmente el plato."""

    return image_caption_system_msg, image_caption_prompt_txt


### Test the prompts on the first recipe
### Step 3.2: Get the prompts with the food name of the first recipe
prompt_template = recipe_data[0]['name']
prompt_image_system, image_caption_prompt = image_caption_prompt_template(prompt_template)


### Step 3.3: Get the test response and print it
response = vision_llm(prompt_image_system, image_caption_prompt, url_image_recipe)
print(response)



### Your Code Here:

### Get captions for each image in the dataset and add to the JSON file (up tp 15 minutes)
### recipe_data is the recipe you loaded in Step 1
for i in range(len(recipe_data)):
    if (i+1)%20 == 0:
        print(f'{i+1} out of {len(recipe_data)} is done')

    ### Step 4.1: Get the caption prompts
    food_name_recipe = recipe_data[i]['name']
    prompt_image_system, image_caption_prompt = image_caption_prompt_template(food_name_recipe)

    ### Step 4.2: Get the response with the prompts
    url_image = f"synthetic_recipe_images/recipe{recipe_data[i]['id']}.png"
    response = vision_llm(prompt_image_system, image_caption_prompt, url_image)

    ### Save the response as another item in the recipe data
    recipe_data[i]['image_description'] = response
print('ALL DONE!')



filename = 'augmented_food_recipe.json'
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(recipe_data, f, indent=4)


### Part 1
### Your Code Here

### Step 1.1: Load the review dataset with variable name user_review_data
file_path = "Synthetic-User-Reviews.json"
with open(file_path, 'r') as file:
    user_review_data = json.load(file)
    

### Step 1.2: Print the first review by key-value pairs
for key, value in user_review_data[0].items():
    print(f"{key} ({type(value)}) : {value}")

### Part 2
### Your Code Here

### Get the image by requesting from the URL
import ast # needed to convert string representation of list to actual python list
import requests

### Convert the list of images of the first user into the actual python list
### Step 1.3: Use ast.literal_eval to convert
images_url = ast.literal_eval(user_review_data[0]['images'])


### Step 1.4: Use the requests.get() method to get the image content
response = requests.get(images_url[0])


### Step 1.5: Write the image content in Step 1.4 to a temporary file 'review_image_placeholder.jpg'
if response.status_code == 200:
    with open("review_image_placeholder.jpg", "wb") as file:
        file.write(response.content)

### Step 1.6: Open the 'review_image_placeholder.jpg', and show th image
Image.open("review_image_placeholder.jpg")

### Your Code Here:
### Prompt template: caption the images with the context of the reviews
def review_context_image_caption_prompt_template(reviews):
    # reviews: the written review content


    ### Step 2.1: Design your prompts
    review_context_image_caption_system_msg = """You are an assistant specialized in describing food photos taken by customers, using the context of their written review to identify what stands out in the image.
    Focus on visual details that are relevant to what the review describes (e.g., presentation, portion size, condition of the food)."""

    review_context_image_caption_prompt_txt = f"""Here is a customer's review of a dish:
"{reviews}"

Describe the food image the customer uploaded, in light of what they wrote in their review. Be concise: 2-3 sentences maximum, focusing on visual details connected to their comments."""


    return review_context_image_caption_system_msg, review_context_image_caption_prompt_txt

### Step 2.2: Get the prompts
test_prompt = user_review_data[0]['text']
prompt_image_caption, prompt_image_text = review_context_image_caption_prompt_template(test_prompt)

### Step 2.3: Get the response by the vision_llm you defined previously
response = vision_llm(prompt_image_caption, prompt_image_text, "review_image_placeholder.jpg")
print(response)



### Your Code Here



### URL Request function with Retry
# Retries up to 10 times, starting at 1s and doubling (1s, 2s, 4s...)
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=10))
def get_data_with_retry(url):
    response = requests.get(url, timeout=5)
    response.raise_for_status() # Must raise error for retry to trigger
    return response

### Start the for loop
for i in range(len(user_review_data)):
    ### Step 3.1: Convert the string to the Python list of image urls
    review_images = ast.literal_eval(user_review_data[i]['images'])

    review_image_captions = []
    if len(review_images) > 0:
        for img_url in review_images:
            try:
                ### Step 3.2: Use get_data_with_retry to get the image_data
                image_data = get_data_with_retry(img_url)
                print("Success!")
            except Exception as e:
                print(f"All retries failed at url {img_url}:", e)
                continue
            image = image_data.content
            with open('review_image_placeholder.jpg', 'wb') as img_file:
                img_file.write(image)

            ### Step 3.3: Get the prompts, get the response, and finally append the response to review_image_captions
            test_prompt = user_review_data[i]['text']
            prompt_image_caption, prompt_image_text = review_context_image_caption_prompt_template(test_prompt)
            review_image = vision_llm(prompt_image_caption, prompt_image_text, "review_image_placeholder.jpg")
            review_image_captions.append(review_image)
            
    ### Append the review_image_captions to the review data
    user_review_data[i]['image_captions'] = review_image_captions
print('ALL DONE!')



filename = 'augmented_user_review.json'
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(user_review_data, f, indent=4)