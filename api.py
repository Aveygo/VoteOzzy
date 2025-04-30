"""

(all endpoints are sessioned)

[GET] /question?answer -> {next_question, rankings}
[GET] /reset

"""

from fastapi import FastAPI
from pydantic import BaseModel

class Politician(BaseModel):
    name: str
    party: str 

class CurrentQuestion(BaseModel):
    policy_name: str
    rankings: list[Politician]


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/question")
async def question(answer:int|None=None) -> CurrentQuestion | None:
    return {"message": "Hello World"}