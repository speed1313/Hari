from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def retrieve_needle(self, haystack: str, retrieval_question: str) -> str:
        pass
