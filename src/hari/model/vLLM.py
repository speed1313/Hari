from hari.model.model import Model, INSTRUCTION_FORMAT
from vllm import LLM, SamplingParams


class VLLM(Model):
    def __init__(self, model_name="google/gemma-3-1b-it"):
        self.model_name = model_name
        self.llm = LLM(model=model_name, tensor_parallel_size=1)
        self.tokenizer = self.llm.get_tokenizer()

    def retrieve_needle(self, haystack: str, retrieval_question: str) -> str:
        prompts = [
            INSTRUCTION_FORMAT.format(
                haystack=haystack, retrieval_question=retrieval_question
            )
        ]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

        outputs = self.llm.generate(prompts, sampling_params)
        return outputs[0].outputs[0].text

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list) -> str:
        return self.tokenizer.decode(tokens)


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
