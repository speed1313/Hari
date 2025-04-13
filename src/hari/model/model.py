from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def retrieve_needle(self, haystack: str, retrieval_question: str) -> str:
        pass


INSTRUCTION_FORMAT = """
You will be given a long document of text. Within the document, there is a single fact.
At the end of the document, there is a question.
You need to answer the question based solely on the document.
You are not allowed to use any external knowledge.

Document:
{haystack}

Question:
{retrieval_question}

Answer:"""
