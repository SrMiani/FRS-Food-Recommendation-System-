from prompt_template import EXAMPLE_RESTAURANT_PARAGRAPH, EXAMPLE_OUTPUT



def restaurant_data_prompt_builder(restaurant_paragraph):
    base_system_msg="""
    Eres un asistente útil que extrae información de un párrafo descriptivo de un restaurante y 
    la convierte en un formato JSON estructurado usando los siguiente campos:
    {{
            "name": "",
            "location": "",
            "type": "",
            "food_style": "",
            "rating": 0.0,
            "price_range": 0,
            "signatures": [],
            "vibe": "",
            "environment": "",
            "shortcomings": []
        }}

    """
    base_user_prompt=f"""
    Tarea:
    Extrae la información del restaurante que aparece abajo y devuélvela en formato JSON, 
    como el del ejemplo de arriba. No inventes ningún dato.
    Si un campo no se menciona en la descripción, usa null (o una lista vacía para signatures/shortcomings).
    No infieras ni adivines información que falte.

    Descripción del restaurante:
    {restaurant_paragraph}

    Ejemplo de entrada:
    {EXAMPLE_RESTAURANT_PARAGRAPH}
    Ejemplo de salida:
    {EXAMPLE_OUTPUT}
    """
    return base_system_msg, base_user_prompt