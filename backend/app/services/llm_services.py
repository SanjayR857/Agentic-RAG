from langchain_ollama import OllamaLLM

class LLMService:
    
    def __init__(self, model_name="llama3.1"):
        self.model = OllamaLLM(model=model_name)

    def get_model(self):
        return self.model
    