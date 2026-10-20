from agents import call_agent
import json

from langgraph.graph import StateGraph, START, END
from typing import Annotated
from typing import TypedDict
from tools import search_restaurants, search_recipes

"""Diseño del workflow (LangGraph)"""



def keep_valid(items:list,valid_id:set) -> list:
    resultado = []
    for r in items:
        if r.get("id") in valid_id:
            resultado.append(r)
    return resultado


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
    "final_recommendations":{},

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
    """Retrieve REAL candidates from the Chroma vector database."""
    print("\n[Phase 2] Retrieving candidates from vector database...")
    query = state["user_input"]
    restaurants = search_restaurants(query, k=20)
    recipes = search_recipes(query, k=20)
    print(f"✓ Retrieved {len(restaurants)} restaurants and {len(recipes)} recipes")
    return {
        "retrieved_restaurants": restaurants,
        "retrieved_recipes": recipes,
        "workflow_step": "candidates_retrieved",
    }



def node_analyze_trends(state: dict) -> dict:
    """Analyze food trends in the retrieved candidates."""
    print("\n[Phase 3a] Analyzing food trends...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    
    user_message = f"""Analyze current food trends in these options:

Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}

Identify 3-5 relevant trends and explain how they align with modern dining culture.
Return JSON: {{"trends": [{{"name": str, "description": str, "relevance": str}}]}}
"""
    
    try:
        response = call_agent("food_trend_analyst", user_message)
        trend_analysis = json.loads(response)
        print(f"✓ Identified {len(trend_analysis.get('trends', []))} trends")
    except Exception as e:
        print(f"⚠ Error analyzing trends: {e}")
        trend_analysis = {"error": str(e)}
    
    state["trend_analysis"] = trend_analysis
    return {"trend_analysis": trend_analysis}




def node_analyze_styles(state: dict) -> dict:
    """Analyze food styles and flavor profiles."""
    print("\n[Phase 3b] Analyzing food styles...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    profile = state["user_profile"]
    
    user_message = f"""
    Analyse cuisine types and  flavor profiles:
    Restaurants : {json.dumps(restaurants,indent=2)}
    Recipes: {json.dumps(recipes,indent=2)}

    User Profile : {json.dumps(profile,indent=2)}
    Respond with a valid JSON object only, with no extra text
    
    """
    
    
    # Fill this in
    
    try:
        response = call_agent("food_style_expert", user_message)
        style_analysis = json.loads(response)
        print(f"✓ Style analysis completed")
    except Exception as e:
        print(f"⚠ Error analyzing styles: {e}")
        style_analysis = {"error": str(e)}
    
    state["style_analysis"] = style_analysis
    return {"style_analysis": style_analysis}




def node_evaluate_nutrition(state: dict) -> dict:
    """Evaluate nutritional aspects and dietary compliance."""
    print("\n[Phase 3c] Evaluating nutrition...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    profile = state["user_profile"]
    
    user_message = f"""Evaluate the nutritional fit of these options:

User Profile: {json.dumps(profile, indent=2)}
Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}

Check dietary restrictions, allergens, and nutritional balance.
Return JSON: {{"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}}
"""
    
    try:
        response = call_agent("nutrition_expert", user_message)
        nutrition_analysis = json.loads(response)
        print(f"✓ Nutrition evaluation completed")
    except Exception as e:
        print(f"⚠ Error evaluating nutrition: {e}")
        nutrition_analysis = {"error": str(e)}
    
    state["nutrition_analysis"] = nutrition_analysis
    return {"nutrition_analysis": nutrition_analysis }



def node_generate_recommendations(state: dict) -> dict:
    
    """Synthesize all analyses into final recommendations, ONLY from retrieved candidates."""
    print("\n[Phase 4] Generating final recommendations...")

    restaurants = state["retrieved_restaurants"][:10]
    recipes = state["retrieved_recipes"][:10]

    user_message = f"""Choose the top 5 restaurants and top 5 recipes for this user.

STRICT RULES:
- You may ONLY choose from the CANDIDATE lists below.
- Copy the "id" exactly. Never invent names or ids.
- If fewer than 5 candidates fit, return fewer.

User Profile: {json.dumps(state['user_profile'], indent=2)}
Trends: {json.dumps(state['trend_analysis'], indent=2)}
Styles: {json.dumps(state['style_analysis'], indent=2)}
Nutrition: {json.dumps(state['nutrition_analysis'], indent=2)}

CANDIDATE RESTAURANTS:
{json.dumps(restaurants, indent=2)}

CANDIDATE RECIPES:
{json.dumps(recipes, indent=2)}

Return JSON:
{{
  "restaurants": [{{"id": str, "name": str, "reasoning": str}}],
  "recipes": [{{"id": str, "name": str, "reasoning": str}}]
}}
Each reasoning: 2-3 sentences.
"""

    try:
        response = call_agent("recommendation_expert", user_message)
        recs = json.loads(response)

        valid_rest = {r["doc_id"] for r in restaurants}
        valid_reci = {r["doc_id"] for r in recipes}
        recs["restaurants"] = keep_valid(recs.get("restaurants", []), valid_rest)
        recs["recipes"] = keep_valid(recs.get("recipes", []), valid_reci)

        print(f"✓ {len(recs['restaurants'])} restaurants, {len(recs['recipes'])} recipes (validated)")
    except Exception as e:
        print(f"⚠ Error generating recommendations: {e}")
        recs = {"error": str(e)}

    return {"final_recommendations": recs, "workflow_step": "complete"}




class WorkflowState(TypedDict):
    user_input: str
    user_profile: dict
    retrieved_restaurants: list
    retrieved_recipes: list
    trend_analysis: dict
    style_analysis: dict
    nutrition_analysis: dict
    final_recommendations: dict
    workflow_step: str


def run_workflow(user_input: str) -> dict:
   
    return workflow.invoke({**INITIAL_STATE,"user_input":user_input})


#Construcción del workflow principal

# Construir el grafo
builder = StateGraph(WorkflowState)

# Fase 1 y 2 — secuenciales
builder.add_node("generate_profile", node_generate_profile)
builder.add_node("retrieve_candidates", node_retrieve_candidates)

# Fase 3 — los 3 nodos "en paralelo"
builder.add_node("analyze_trends", node_analyze_trends)
builder.add_node("analyze_styles", node_analyze_styles)
builder.add_node("evaluate_nutrition", node_evaluate_nutrition)

# Fase 4 — síntesis
builder.add_node("generate_recommendations", node_generate_recommendations)

# Conexiones secuenciales
builder.add_edge(START, "generate_profile")
builder.add_edge("generate_profile", "retrieve_candidates")

# Paralelización
def to_parallel(state):
    return ["analyze_trends", "analyze_styles", "evaluate_nutrition"]

builder.add_conditional_edges("retrieve_candidates", to_parallel)

# Los 3 confluyen en el nodo de síntesis
builder.add_edge("analyze_trends", "generate_recommendations")
builder.add_edge("analyze_styles", "generate_recommendations")
builder.add_edge("evaluate_nutrition", "generate_recommendations")

builder.add_edge("generate_recommendations", END)

workflow = builder.compile()