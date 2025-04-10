import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import weave

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


def retrieve_needle(haystack: str, retrieval_question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
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
        max_tokens=1024,
        temperature=0.0,
    )
    return response.choices[0].message.content


def judge_retrieval(retrieval: str, needle: str, question: str) -> int:
    """
    Judge the retrieval between 1 to 10
    """
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": "You are a helpful AI Judger"},
            {
                "role": "user",
                "content": f"""Below is a fact and a question:
                Fact: {needle},
                Question: {question},
                And below is a retrieval from the document: {retrieval},
                Please judge the retrieval based on the fact and question.
                Give a score between 1 to 5, where 1 means the retrieval is completely wrong and 5 means the retrieval is completely correct. Don't give any explanation.""",
            },
        ],
        max_tokens=128,
        temperature=0.0,
    )
    score = int(response.choices[0].message.content.strip())
    print(f"Score: {score}")
    assert score >= 1 and score <= 5, "Score should be between 1 and 5"
    return score


def test_judge_retrieval():
    # only do test if the env variable is set
    if not os.getenv("AZURE_OPENAI_KEY"):
        print("Skipping test_judge_retrieval because AZURE_OPENAI_KEY is not set")
        return
    question = "京都でおすすめの観光地はどこですか？"
    needle = "京都でおすすめの観光地は、ロームシアター京都の３階にあるラウンジです。"
    retrieval = "京都でおすすめの観光地は、ロームシアター京都の３階にあるラウンジです。"
    assert judge_retrieval(retrieval, needle, question) == 5
    retrieval = "ロームシアター京都の３階にあるラウンジです。"
    assert judge_retrieval(retrieval, needle, question) == 5
    retrieval = "金閣寺です。"
    assert judge_retrieval(retrieval, needle, question) == 1
    retrieval = "ロームシアター京都です。"
    score = judge_retrieval(retrieval, needle, question)
    assert score >= 1 and score < 5, "Score should be between 1 and 5"
