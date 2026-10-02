import json
from pydantic import ValidationError
from extract import load_restaurant_data
from prompt_builders import restaurant_data_prompt_builder, JSON_auto_repair_prompt_builder
from models import Restaurant
from llm_model import llm_model_call, safe_llm_call


restaurant_list = load_restaurant_data()
#base_system_msg, base_user_prompt = restaurant_data_prompt_builder(restaurant_paragraph)


structured_data_list = []
restaurants_wrong_validation=[]
for idx,restaurant_paragraph in enumerate(restaurant_list):
    base_system_msg,base_userr_prompt=restaurant_data_prompt_builder(restaurant_paragraph)
    text_response = safe_llm_call(llm_model_call, base_system_msg, base_userr_prompt)
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
            text_response = safe_llm_call(llm_model_call, auto_repair_system_msg, auto_repair_prompt)
            contador_intentos += 1
            if restaurant_paragraph not in restaurants_wrong_validation:
                restaurants_wrong_validation.append(restaurant_paragraph)
    

print ("All restaurants processed. Total:", len(structured_data_list))
print ("Restaurants with validation issues:", len(restaurants_wrong_validation))

structured_data_list_json = [restaurant.model_dump() for restaurant in structured_data_list]


for i, restaurant in enumerate(structured_data_list_json):
    restaurant['id'] = 1000001 + i  # Assign a unique ID starting from 1
   # structured_data_list_json[i] = restaurant  # Update the list with the new dictionary


filename = "structured_restaurant_data.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(structured_data_list_json, f, ensure_ascii=False, indent=4)