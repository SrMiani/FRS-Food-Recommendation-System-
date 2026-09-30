from dotenv import load_dotenv
import os
from openai import OpenAI
import time

from pydantic import ValidationError
from extract import load_restaurant_data
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


def safe_llm_call(system_msg,prompt_txt,max_retries=3):
    for i in range(max_retries):
        try:
            return llm_model_call(system_msg, prompt_txt)
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

restaurant_list = load_restaurant_data()
#base_system_msg, base_user_prompt = restaurant_data_prompt_builder(restaurant_paragraph)


structured_data_list = []
restaurants_wrong_validation=[]
for idx,restaurant_paragraph in enumerate(restaurant_list):
    base_system_msg,base_userr_prompt=restaurant_data_prompt_builder(restaurant_paragraph)
    text_response = safe_llm_call(base_system_msg, base_userr_prompt)
    contador_intentos= 0
    while contador_intentos < 3:
        try:
            texto_limpio = text_response.strip()
            # Remove any leading or trailing characters that are not part of the JSON
            if texto_limpio.startswith("```"):
                texto_limpio = texto_limpio.strip("`")
                texto_limpio = texto_limpio.removeprefix("json").strip()
            restaurant_data = Restaurant.model_validate_json(texto_limpio)
            print(f"Restaurant {idx}: {restaurant_data.name} validated successfully.")
            structured_data_list.append(restaurant_data)
            break
        except ValidationError as e:
            print(f"Validation error for restaurant {idx}: {e.json()}")
            error_msg = e.json()
            auto_repair_system_msg, auto_repair_prompt = JSON_auto_repair_prompt_builder(text_response, error_msg)
            text_response = safe_llm_call(auto_repair_system_msg, auto_repair_prompt)
            contador_intentos += 1
            if restaurant_paragraph not in restaurants_wrong_validation:
                restaurants_wrong_validation.append(restaurant_paragraph)
    

print ("All restaurants processed. Total:", len(structured_data_list))
print ("Restaurants with validation issues:", len(restaurants_wrong_validation))










#print(llm_model_call("You are a helpful assistant.", 
                     #"Which place is warmer in winter? Hawaii or Greenland?"))


    
