import base64
import json

from dotenv import load_dotenv
import os
from openai import OpenAI
import time

from pydantic import ValidationError

from prompt_builders import restaurant_data_prompt_builder, JSON_auto_repair_prompt_builder
from models import Restaurant



load_dotenv()



def llm_model_call(system_msg, prompt_txt):
    # system_msg: the system message given to the LLM
    # prompt_txt: the user prompt
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    output_text = response.choices[0].message.content

    return output_text


def vision_model_call(image_path, prompt_txt,system_msg):
    # image_path: path to the image file   
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    messages=[
        {"role": "system", "content": system_msg},

        {
            "role": "user", 
            "content": [
                {"type": "text", "text": prompt_txt},
                {
                    "type": "image_url", 
                    "image_url":{"url": f"data:image/png;base64,{encoded_image}"}
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        
    )

    output_text = response.choices[0].message.content

    return output_text











def safe_llm_call(func,*args, max_retries=3,**kwargs):
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            time.sleep(2)
    return "Failed after retries"
    


#restaurant_paragraph = restaurant_list[1]


#text_response=safe_llm_call(base_system_msg, base_user_prompt)

#print("Restaurant paragraph:")
#print(restaurant_paragraph)

#print("LLM Response:")
#print(text_response)

#testing validation of the JSON response against the Restaurant model
#try:
 #   restaurant_data = Restaurant.model_validate_json(text_response)
  #  print("Validated Restaurant Data:")
  #  print(f"{restaurant_data.name}")
#except ValidationError as e:
 #   print("Failed to parse JSON into Restaurant model.")
  #  print(f"Error:  {e.json()}")














#print(llm_model_call("You are a helpful assistant.", 
                     #"Which place is warmer in winter? Hawaii or Greenland?"))


    
