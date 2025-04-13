from hari.model.model import Model
from vllm import LLM, SamplingParams


class VLLM(Model):
    def __init__(self, model_name="google/gemma-3-1b-it"):
        self.model_name = model_name
        self.llm = LLM(model=model_name, tensor_parallel_size=4)

    def retrieve_needle(self, haystack: str, retrieval_question: str) -> str:
        prompts = [
            f"""You will be given a long document of text. Within the document, there is a single sentence that is needed to answer the question at the end.
            Read the entire document carefully, then answer the question as accurately as possible. Please do not include any additional information or context outside of the passage.

            Document:
            { haystack }

            Question:
            { retrieval_question }

            Answer:""",
        ]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

        outputs = self.llm.generate(prompts, sampling_params)
        return outputs[0].outputs[0].text


def test_vllm():
    model = VLLM()
    assert (
        model.retrieve_needle(
            "京都でおすすめの観光地は、ロームシアター京都の３階にあるラウンジです。",
            "京都でおすすめの観光地はどこですか？",
        )
        == "ロームシアター京都の３階にあるラウンジです。"
    )


if __name__ == "__main__":
    model = VLLM()
    print(
        model.retrieve_needle(
            "京都でおすすめの観光地は、ロームシアター京都の３階にあるラウンジです。",
            "京都でおすすめの観光地はどこですか？",
        )
    )
