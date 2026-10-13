from agents import call_agent

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




