import os
from openai import AzureOpenAI
from hari.model.model import Model


class GPT4o(Model):
    def __init__(self, model_name="gpt-4o-2024-11-20"):
        self.model_name = model_name
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2023-05-15",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    def retrieve_needle(self, haystack: str, retrieval_question: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct",
                },
                {"role": "user", "content": f"Below is a document: {haystack}"},
                {
                    "role": "user",
                    "content": f"{retrieval_question} Don't give information outside the document or repeat your findings",
                },
            ],
            max_tokens=256,
            temperature=0.0,
        )
        return response.choices[0].message.content


def test_retrieve_needle():
    # only do test if the env variable is set
    if not os.getenv("AZURE_OPENAI_KEY"):
        print("Skipping test_retrieve_needle because AZURE_OPENAI_KEY is not set")
        return
    model = Model()
    haystack = "京都でおすすめの観光地は、ロームシアター京都の３階にあるラウンジです。"
    retrieval_question = "京都でおすすめの観光地はどこですか？"
    assert "ロームシアター京都" in model.retrieve_needle(haystack, retrieval_question)
