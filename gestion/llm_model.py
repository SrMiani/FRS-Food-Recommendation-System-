from dotenv import load_dotenv
import os
from openai import OpenAI
import time



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
    


print(llm_model_call("You are a helpful assistant.", "Which place is warmer in winter? Hawaii or Greenland?"))


    