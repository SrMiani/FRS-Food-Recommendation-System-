from pydantic import BaseModel,Field,ValidationError
from typing import List, Optional


#Definimos el modelo de datos para un restaurante
class Restaurant(BaseModel):
    name: str = Field(..., description="Nombre del restaurante")
    location: str = Field(..., description="Ubicación del restaurante")
    type: str = Field(..., description="Tipo de restaurante")
    food_style: str = Field(..., description="Estilo de comida")
    rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Calificación del restaurante (0.0 a 5.0)")
    price_range: Optional[int] = Field(None, ge=1, le=5, description="Rango de precios (1 a 5)")
    signatures: List[str] = Field(default_factory=list, description="Platos o bebidas destacadas")
    vibe: Optional[str] = Field(None, description="Ambiente o sensación del lugar")
    environment: Optional[str] = Field(None, description="Descripción del entorno o ambiente")
    shortcomings: List[str] = Field(default_factory=list, description="Deficiencias o aspectos negativos")