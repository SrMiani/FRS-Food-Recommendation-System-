from fastapi import FastAPI
from pydantic import BaseModel
from workflow import run_workflow


app=FastAPI(title="PlatePicker API")

class RecommendRequest(BaseModel):
    user_input:str


@app.post("/recommend")
def recommend(req:RecommendRequest):
    result = run_workflow(req.user_input)
    return{
        "profile":result["user_profile"],
        "recommendations": result["final_recommendations"],

    }

@app.get("/version")
def version():
    return{"version":"1.0"}




