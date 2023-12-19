from fastapi import FastAPI
import nn_module  # Модуль с нейронной сетью
from pydantic import BaseModel

app = FastAPI()


#class PostGet(BaseModel):
#    text: str
#    result: str


@app.get("/prediction/") #, response_model=PostGet
def predict_postman(text: str = 'None'):
    # Обработка текста с использованием нейронной сети
    result = nn_module.predict(text)

    return f'Ваш Запрос - {text}', result