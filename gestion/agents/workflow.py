from agents import call_agent
import json

"""Diseño del workflow (LangGraph)"""

#Definimos la estructura del state como diccionario
INITIAL_STATE={
    #Input
    "user_input":"",

    #Fase1: Analsis de usuario
    "user_profile":{},

    #Fase2 RAG
    "retrieved_restaurants":[],
    "retrieved_recipes":[],

    #Fase3 Analisis en paralelo
    "trend_analysis":{},
    "style_analysis":{},
    "nutrition_analysis":{},

    #Fase4 Sintesis final
    "final_recommendation":{},

    #Metadata
    "workflow_step":"start"
}

print(f"Estructura State definida con {len(INITIAL_STATE)} campos:")
for key in INITIAL_STATE:
    print(f"   - {key}")


#Primer nodo
def node_generate_profile(state:dict)->dict:
    """Generate user profile from input data."""
    print("\n[Phase 1] Generating user profile...")

    user_message=f"""Analyse this user data and create a comprehensive profile:
    
    {state['user_input']}

    Provide output in JSON format with these keys:
- favorite_cuisines (list)
- dietary_restrictions (list)
- dining_occasions (list)
- price_range (string)
- adventurousness_score (1-10)
- flavor_preferences (list)
- summary (string)
"""
    
    
    try:
        response = call_agent("user_profile_generator",user_message)
        user_profile=json.loads(response)
        print(f"User profile generated: {user_profile.get('summary','No summary')}")
    except Exception as e:
        print(f"Error generated {e}")
        user_profile={"error" : str(e)}

    state["user_profile"] = user_profile
    state["workflow_step"] = "profile_generated"
    return state


#Segundo nodo
def node_retrieve_candidates(state: dict) -> dict:
    """Retrieve restaurant and recipe candidates from vector database."""
    print("\n[Phase 2] Retrieving candidates from vector database...")
    
    profile = state["user_profile"]
    
    user_message = f"""Based on this user profile:
{json.dumps(profile, indent=2)}

Simulate retrieving top 20 restaurants and top 20 recipes from a vector database.

Return JSON with two arrays:
- restaurants: [{{"name": str, "cuisine": str, "price": str, "rating": float, "description": str}}]
- recipes: [{{"name": str, "cuisine": str, "difficulty": str, "prep_time": str, "description": str}}]

Make the results realistic and diverse.
"""
    
    try:
        response = call_agent("rag_retriever", user_message)
        retrieved_data = json.loads(response)
        restaurants = retrieved_data.get("restaurants", [])
        recipes = retrieved_data.get("recipes", [])
        print(f"✓ Retrieved {len(restaurants)} restaurants and {len(recipes)} recipes")
    except Exception as e:
        print(f"⚠ Error retrieving candidates: {e}")
        restaurants, recipes = [], []
    
    state["retrieved_restaurants"] = restaurants
    state["retrieved_recipes"] = recipes
    state["workflow_step"] = "candidates_retrieved"
    return state



