import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class Config:
    def __init__(self):
        pass
    
    def getString(self, key:str) -> str:
        return os.getenv(key)
    
    def getLLM(self, model_name:str="gpt-3.5-turbo", temperature:float=0.0):
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=self.getString("OPENAI_API_KEY"),
        )
        return llm