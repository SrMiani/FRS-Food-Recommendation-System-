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


def JSON_auto_repair_prompt_builder(candidate_json_string,error_message):
    auto_repair_system_msg = """
    Eres un asistente especializado en corregir JSON mal formado según un esquema específico. 
    Se te dará un JSON con errores y el mensaje de error de validación. 
    Tu trabajo es corregir el JSON para que cumpla el esquema, sin cambiar la información real que ya contiene.
    """
    auto_repair_prompt = f"""
    Task: El siguiente JSON no pasó la validación. Corrígelo basándote en el error indicado.

    JSON con errores:
    {candidate_json_string}

    Error de validación:
    {error_message}

    Devuelve únicamente el JSON corregido, sin explicaciones adicionales.
    """
    return auto_repair_system_msg, auto_repair_prompt

