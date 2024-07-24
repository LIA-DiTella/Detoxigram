# Facade

from Analyzer import Analyzer
from Detoxifier import Detoxifier
from Explainer import Explainer
from Dataviz.ToxicityDataviz import ToxicityDataviz

class Toxicity:
    def __init__(self, model_name: str, llm, detoxigramer):
        self.analyzer = Analyzer(model_name)
        self.detoxifier = Detoxifier(model_name)
        self.explainer = Explainer(llm, detoxigramer)
        self.dataviz = ToxicityDataviz()
    
    def analyze_toxicity(self, messages: list) -> list:
        return [self.analyzer.analyze(message) for message in messages]
    
    def detoxify_message(self, message: str) -> str:
        return self.detoxifier.detoxify_message(message)
    
    def explain_toxicity(self, messages: list, escala: str, toxicity: int) -> list:
        return self.explainer.explain_toxicity(messages, escala, toxicity)
    
    def dimensions_toxicity(self, conversation : str, conversation_name : str, toxicity : int):
        return self.dataviz.get_toxicity_dimensions(conversation, conversation_name, toxicity)
