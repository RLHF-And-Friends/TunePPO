import typing as tp

from functools import partial
from ppotune.data.sets.multi_hop import MultiHopDataset, MultiHopProblem, MultihopTransform


TWO_HOP_SYSTEM_PROMPT = '''You are a chain-of-thought language model. When the user asks a question you MUST reply in the **exact** structure below ­– nothing more, nothing less
<think>
<question>first self-generated sub-question</question><answer>answer to the first sub-question</answer>
<question>second self-generated sub-question</question><answer>answer to the second sub-question</answer>
</think>
<answer>final answer to the user’s original question</answer>

Mandatory rules
1. Produce **exactly two** sub-questions, each immediately followed by its answer, both wrapped in the indicated tags.
2. All four inner tags (<question> / <answer>) live **inside** a single <think> … </think> block.
3. After the </think> tag, output one—and only one—final answer, wrapped in its own outer <answer> … </answer> tag.
4. Do not reveal any additional text, commentary, or tags outside those shown above.
5. Preserve the tag names, their order, and the line breaks precisely as specified.
'''

class TwoHopTransform(MultihopTransform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> MultiHopProblem:
        question = sample["generated_question"]
        answers = []
        answers.append(sample["second_entity_aliases"])
        answers.append(sample["third_entity_aliases"])
        final_answer = sample["third_entity_aliases"]

        return MultiHopProblem(
            question=question,
            answers=answers,
            final_answer=final_answer
        )

two_hop_dataset = partial(
    MultiHopDataset,
    sample_transform=TwoHopTransform(),
    system_prompt=TWO_HOP_SYSTEM_PROMPT,
)

