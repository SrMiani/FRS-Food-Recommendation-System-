
import os
import json
from typing import TypedDict, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")

client = OpenAI()
MODEL="gpt-5-mini"

#Configuracion inicial de agentes(En inglés para mejorar la eficiencia del modelo)
agent_configs = {
    "user_profile_generator":{
        "role":"User Profile Generator",
        "goal":"Analyze user restaurant visit history and social media posts to create a comprehensive profile.",
        "backstory":"""You are an expert user behavior analyst with 10 years of expereience in the food industry.
        You excel at identifying patterns in dining behavior and building rich user profiles."""
    },
    "rag_retriever": {
        "role": "RAG Retriever",
        "goal": "Query multimodal vector databases to retrieve relevant restaurants and recipes.",
        "backstory": """You are a data retrieval specialist with expertise in vector databases and semantic search."""
    },
    "food_trend_analyst": {
        "role": "Food Trend Analyst",
        "goal": "Identify current food trends and emerging dining concepts.",
        "backstory": """You are a culinary journalist who has spent 15 years covering food culture across global markets."""
    },
    "food_style_expert": {
        "role": "Food Style Expert",
        "goal": "Analyze cuisine types and flavor profiles to match user preferences.",
        "backstory": """You are a trained chef and culinary anthropologist with expertise in global cuisines."""
    },
    "nutrition_expert": {
        "role": "Nutrition Expert",
        "goal": "Evaluate nutritional content and ensure dietary compliance.",
        "backstory": """You are a registered dietitian with 10 years of clinical experience."""
    },
    "recommendation_expert": {
        "role": "Recommendation Expert",
        "goal": "Synthesize insights from all agents into final recommendations.",
        "backstory": """You are a recommendation systems architect with experience in personalization engines."""
    }
}


print ("Agents configuration loaded succesfully")


def call_agent(agent_key:str,user_message:str)->str:
    """Call an agent with a specific message and return its response."""
    config = agent_configs[agent_key]

    system_prompt = f"""You are a {config['role']}.

    Your goal: {config['goal']}
    Your background: {config['backstory']}

Respond with strcutured, actionable output.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system","content":system_prompt},
            {"role": "user","content":user_message}
        ]
    )

    return response.choices[0].message.content







