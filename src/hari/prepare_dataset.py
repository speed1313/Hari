from datasets import load_dataset
import numpy as np


def build_haystack(ds, context_length_max: int) -> str:
    haystack = ""
    for example in ds:
        if len(haystack) > context_length_max:
            break
        text = example["text"]
        haystack += text.strip() + "\n"
    return haystack


def insert_needle_into_haystack(
    haystack: str, needle: str, context_length_max: int, position: int
) -> (str, int):
    """
    Insert the needle into the haystack at the specified position.
    If the position is in the middle of a sentence, the needle will be inserted at the beginning of the sentence.
    """

    if position > (context_length_max - len(needle)):
        new_haystack = (
            haystack[: context_length_max - (len(needle) + 1)] + "\n" + needle
        )
        return new_haystack, (context_length_max - len(needle) - 1)

    truncated_haystack = haystack[: context_length_max - len(needle)]
    period_tokens = set(["。", ".", "\n", "!", "?"])
    insert_at = 0
    for i in range(position, 0, -1):
        if truncated_haystack[i] in period_tokens:
            insert_at = i + 1
            break
    if insert_at + len(needle) > context_length_max:
        raise ValueError("Needle exceeds context length max.")
    new_haystack = (
        truncated_haystack[:insert_at] + needle + truncated_haystack[insert_at:]
    )

    new_haystack = new_haystack[:context_length_max]
    return new_haystack, insert_at


def prepare_various_haystack(
    haystack: str,
    needle: str,
    context_length_max: int,
    context_length_min: int,
    interval: int,
) -> list:
    """
    Prepare various haystack with different context length and interval.
    """
    haystacks = []
    for i in range(context_length_min, context_length_max, interval):
        new_haystack, position = insert_needle_into_haystack(
            haystack, needle, context_length_max, i
        )
        haystacks.append((new_haystack, position))
    return haystacks


def prepare_haystacks_across_lengths_and_positions(
    ds, needle: str, lengths: list = [1024, 2048, 4096, 8192, 16384], depth: int = 5
) -> list:
    """
    Create haystacks of various lengths with needle inserted at various relative positions.
    """
    relative_positions = np.linspace(0, 1, depth)  # y-axis
    all_haystacks = []

    for length in lengths:
        haystack = build_haystack(ds, length)
        for rel_pos in relative_positions:
            abs_pos = int(length * rel_pos)

            new_haystack, insert_at = insert_needle_into_haystack(
                haystack, needle, length, abs_pos
            )
            print(
                f"Haystack Length: {len(new_haystack)}, max_length: {length}, relative_position: {rel_pos}, absolute_position: {insert_at}"
            )
            all_haystacks.append(
                {
                    "length": length,
                    "depth": rel_pos,
                    "absolute_position": insert_at,
                    "haystack": new_haystack,
                }
            )

    return all_haystacks


# def test_insert_needle_into_haystack():
#     import pytest

#     haystack = (
#         "Today is sunny.\nThe quick brown fox jumps over the lazy dog.\nToday is rainy."
#     )
#     needle = "This is needle."
#     context_length_max = 60
#     result = insert_needle_into_haystack(haystack, needle, context_length_max, 0)
#     assert result == (
#         "This is needle.\nToday is sunny.\nThe quick brown fox jumps ov",
#         0,
#     )

#     result = insert_needle_into_haystack(haystack, needle, context_length_max, 10)
#     assert result == (
#         "This is needle.\nToday is sunny.\nThe quick brown fox jumps ov",
#         0,
#     )

#     result = insert_needle_into_haystack(haystack, needle, context_length_max, 25)
#     assert result == (
#         "Today is sunny.\nThis is needle.\nThe quick brown fox jumps ov",
#         16,
#     )

#     with pytest.raises(ValueError, match="Position exceeds haystack length."):
#         insert_needle_into_haystack(haystack, needle, context_length_max, 80)


if __name__ == "__main__":
    ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
    ds = ds.shuffle(seed=42)
    question = "京都でおすすめの観光地はどこですか？"
    needle = "京都でおすすめの観光地は、ロームシアター京都の３階にあるラウンジです。"
    # Prepare haystacks across lengths and positions
    all_haystacks = prepare_haystacks_across_lengths_and_positions(ds, needle)
    import json

    with open("haystacks.json", "w") as f:
        json.dump(all_haystacks, f, ensure_ascii=False, indent=4)
