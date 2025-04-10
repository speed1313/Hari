import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import weave

weave.init("Hari")
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


if __name__ == "__main__":
    from hari.prepare_dataset import prepare_haystacks_across_lengths_and_positions
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
    ds = ds.shuffle(seed=42)
    question = "京都でおすすめの観光地はどこですか？"
    needle = "京都でおすすめの観光地は、ロームシアター京都の３階にあるラウンジです。"
    ground_truth = "ロームシアター京都の３階にあるラウンジ"
    # Prepare haystacks across lengths and positions
    all_haystacks = prepare_haystacks_across_lengths_and_positions(
        ds, needle, lengths=[1024, 2048]
    )

    retrieval_question = "京都でおすすめの観光地はどこですか？"
    result_info = {}

    import unicodedata
    import numpy as np

    lengths = sorted(set([info["length"] for info in all_haystacks]))
    depths = sorted(set([info["depth"] for info in all_haystacks]))

    # Generate 2D accuracy matrix
    z_scores = np.zeros((len(depths), len(lengths)))
    for i, depth in enumerate(depths):
        for j, length in enumerate(lengths):
            haystack = None
            for info in all_haystacks:
                if info["length"] == length and info["depth"] == depth:
                    haystack = info["haystack"]
                    break
            retrieved = retrieve_needle(haystack, question)
            print(f"Haystack Length: {length}, Depth: {depth}, Retrieved: {retrieved}")
            # Normalize the retrieved string
            retrieved = unicodedata.normalize("NFKC", retrieved)
            # Normalize the needle string
            ground_truth = unicodedata.normalize("NFKC", ground_truth)
            accuracy = int(ground_truth in retrieved)
            z_scores[i, j] = accuracy
