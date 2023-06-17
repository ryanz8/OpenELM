import json
from abc import ABC
from dataclasses import dataclass
from typing import List

import numpy as np

from openelm.constants import SRC_PATH


def get_image_target(name: str) -> np.ndarray:
    if name == "circle":
        target = np.zeros((32, 32, 3))
        for y in range(32):
            for x in range(32):
                if (y - 16) ** 2 + (x - 16) ** 2 <= 100:  # a radius-10 circle
                    target[y, x] = np.array([255, 255, 0])
    else:
        raise NotImplementedError(f"Image target {name} not implemented")
    return target


IMAGE_SEED: str = """
def draw():
\tpic = np.zeros((32, 32, 3))
\tfor x in range(2, 30):
\t\tfor y in range(2, 30):
\t\t\tpic[x, y] = np.array([0, 0, 255])
\treturn pic
"""

NULL_SEED: str = ""


@dataclass
class ToyPromptTask:
    base_template = "{few_shot_examples}\n{instruction_str} the word {target} {n_repetitions} times:"
    input_variables = [
        "few_shot_examples",
        "target",
        "instruction_str",
        "n_repetitions",
    ]

    target = "hello"
    instruction_str = "Repeat"

    mutation_instructions = [
        """Q: What is a synonym for happy?
A: Cheerful

Q: What is a synonym for sad?
A: Melancholy

Q: What is a synonym for alter?
A: Adjust

Q: What is a synonym for finish?
A: End

Q: What is a synonym for {instruction_str}?
A:"""
    ]

    def create_few_shot_examples(self, instruction_str):
        return f"""{instruction_str} the word {self.target} 2 times: {self.target} {self.target}
{instruction_str} the word {self.target} 3 times: {self.target} {self.target} {self.target}
{instruction_str} the word {self.target} 4 times: {self.target} {self.target} {self.target} {self.target}"""


@dataclass
class PromptTask(ABC):
    base_template: str
    input_variables: List[str]
    generation_instruction: str
    mutation_instructions: List[str]
    few_shot_template: str

    def get_random_data(self, n_examples):
        assert n_examples <= len(
            self.input_list
        ), "n_examples is larger than available data size"
        indices = np.random.choice(len(self.input_list), size=n_examples, replace=False)
        return [self.input_list[idx] for idx in indices], [
            self.output_list[idx] for idx in indices
        ]

    def create_few_shot_examples(self, n_examples):
        few_shot_examples = ""
        sampled_inputs, sampled_outputs = self.get_random_data(n_examples)

        for input_str, output_str in zip(sampled_inputs, sampled_outputs):
            few_shot_examples += self.few_shot_template.format(
                input_str=input_str, output_str=output_str
            )

        return few_shot_examples


@dataclass
class QAPromptTask(PromptTask):
    base_template = """Instruction: {instruction_str}
Input: {input_str}
Output: {output_str}"""

    input_variables = [
        "instruction_str",
        "input_str",
        "output_str",
    ]

    generation_instruction = """I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n{few_shot_examples}\nThe instruction was to """

    mutation_instructions = [
        """Generate a new instruction based on the old instruction that keeps the semantic meaning.

Old instruction: {instruction_str}

New instruction: """,
        """Rewrite this instruction to be more polite.

Old instruction: {instruction_str}

New instruction: """,
        """Rewrite this instruction to be more forceful.

Old instruction: {instruction_str}

New instruction: """,
    ]

    #     evaluation_instruction = """Instruction: {instruction_str}
    # Input: {input_str}
    # Output: {output_str}"""

    few_shot_template = "Input: {input_str}\nOutput: {output_str}\n\n"

    def load_data(self, data_path):
        # Load the data
        with open(SRC_PATH / data_path) as f:
            data = json.load(f)

        # Initialize lists
        self.input_list = []
        self.output_list = []

        # Iterate over examples
        for key, value in data["examples"].items():
            self.input_list.append(value["input"])
            self.output_list.append(value["output"])

        assert len(self.input_list) == len(self.output_list)


@dataclass
class AnimalPromptTask(QAPromptTask):
    def __init__(self):
        self.load_data("environments/prompt/datasets/raw/induce/larger_animal.json")


@dataclass
class AntonymPromptTask(QAPromptTask):
    def __init__(self):
        self.load_data("environments/prompt/datasets/raw/induce/antonyms.json")


@dataclass
class COTPromptTask(PromptTask):
    base_template = """Instruction: Answer the following question.
Q: {input_str}
A: Let's {instruction_str} {output_str}"""

    input_variables = [
        "instruction_str",
        "input_str",
        "output_str",
    ]

    generation_instruction = """Here are some math problems I did with my student.\n{few_shot_examples}\nWe encountered a hard one. They asked me how to start and I said, \"Let's"""

    mutation_instructions = [
        """Generate a new instruction based on the old instruction that keeps the semantic meaning.

Old instruction: {instruction_str}

New instruction: """,
        """Rewrite this instruction to be more polite.

Old instruction: {instruction_str}

New instruction: """,
        """Rewrite this instruction to be more forceful.

Old instruction: {instruction_str}

New instruction: """,
    ]

    few_shot_template = "Q: {input_str}\nA: {output_str}\n\n"

    def __init__(self):
        self.input_list = []
        self.output_list = []

        # load CoT dataset
        with open(
            SRC_PATH / "environments/prompt/datasets/cot_dataset/addsub.csv", "r"
        ) as file:
            for line in file:
                row = line.strip().split(",")
                self.input_list.append(row[0])
                self.output_list.append(row[1])

        assert len(self.input_list) == len(self.output_list)


@dataclass
class ImageMutationPromptTask:
    base_template = """{instruction_str}"""

    input_variables = [
        "instruction_str",
    ]

    fitness_template = """
import math
import numpy as np

def draw():
    \"\"\"
    {instruction_str}

    Returns:
        np.ndarray: the image
    \"\"\"
    pic = np.zeros((32, 32, 3))
    {program_str}
"""

    instruction_str = "Draws a yellow circle."
