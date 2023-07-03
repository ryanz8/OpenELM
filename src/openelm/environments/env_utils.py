import json
from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List

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


def color_circle(x, y, x0, y0, radius):
    """Colors the pixel at (x, y) yellow if it is within a circle of radius `radius` centered at (x0, y0)."""
    if (y - y0) ** 2 + (x - x0) ** 2 <= radius**2:
        return np.array([255, 255, 0])
    else:
        return np.array([0, 0, 0])


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
    initial_prompts = []

    def create_few_shot_examples(self, instruction_str):
        return f"""{instruction_str} the word {self.target} 2 times: {self.target} {self.target}
{instruction_str} the word {self.target} 3 times: {self.target} {self.target} {self.target}
{instruction_str} the word {self.target} 4 times: {self.target} {self.target} {self.target} {self.target}"""


@dataclass
class PromptTask(ABC):
    data: Dict[str, Dict[str, list]] = field(default_factory=dict)

    def get_random_data(self, n_examples, dataset="generation"):
        """
        Returns n_examples random examples from the dataset.

        Args:
            n_examples: number of examples to return. If -1, returns all examples.
            dataset: which key in self.data to sample from ("generation" or "eval")

        Returns:
            input_list: list of input strings
            output_list: list of output strings
        """
        assert n_examples <= len(
            self.data[dataset]["input"]
        ), "n_examples is larger than available data size"
        if n_examples == -1:
            n_examples = len(self.data[dataset]["input"])
        indices = np.random.choice(
            len(self.data[dataset]["input"]), size=n_examples, replace=False
        )
        return [self.data[dataset]["input"][idx] for idx in indices], [
            self.data[dataset]["output"][idx] for idx in indices
        ]

    def create_few_shot_examples(self, n_examples, dataset="generation"):
        few_shot_examples = ""

        sampled_inputs, sampled_outputs = self.get_random_data(n_examples, dataset)

        for input_str, output_str in zip(sampled_inputs, sampled_outputs):
            few_shot_examples += self.few_shot_template.format(
                input_str=input_str, output_str=output_str
            )

        return few_shot_examples


@dataclass
class QAPromptTask(PromptTask):
    base_template: str = """Instruction: {instruction_str}
Input: {input_str}
Output: {output_str}"""

    input_variables: List[str] = field(
        default_factory=lambda: [
            "instruction_str",
            "input_str",
            "output_str",
        ]
    )

    generation_instruction: str = """I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n{few_shot_examples}\nThe instruction was to """

    mutation_instructions: List[str] = field(
        default_factory=lambda: [
            """Generate a new instruction based on the old instruction that keeps the semantic meaning.

Old instruction: Generate a list of five synonyms for the input word.
New instruction: Create a five-item list of synonyms for the word given.

Old instruction: Rewrite the input sentence so the last words form a rhyming couplet.
New instruction: Rephrase the provided sentence so it concludes with a rhyming pair of words.

Old instruction: Analyze the input text and output the primary emotion it conveys.
New instruction: Examine the given text and identify the main emotional tone it expresses.

Old instruction: {instruction_str}
New instruction: """,
            """Rewrite the instruction to be more polite.

Old instruction: Generate a list of five synonyms for the input word.
New instruction: Would you kindly generate a list of five synonyms for the word provided?

Old instruction: Rewrite the input sentence so the last words form a rhyming couplet.
New instruction: Could you please transform the given sentence so that it ends in a rhyming couplet?

Old instruction: Analyze the input text and output the primary emotion it conveys.
New instruction: Could you kindly analyze the text and determine the main emotion it communicates?

Old instruction: {instruction_str}
New instruction: """,
            """Rewrite the instruction to be more forceful.

Old instruction: Generate a list of five synonyms for the input word.
New instruction: You must produce a list of five synonyms for the provided word, no exceptions.

Old instruction: Rewrite the input sentence so the last words form a rhyming couplet.
New instruction: Transform the given sentence into a rhyming couplet immediately.

Old instruction: Analyze the input text and output the primary emotion it conveys.
New instruction: Conduct a swift analysis of the text and determine the dominant emotion it communicates.

Old instruction: {instruction_str}
# New instruction: """,
            # """Rewrite the instruction to be more concise.
            # Old instruction: Generate a list of five synonyms for the input word.
            # New instruction: List five synonyms for the word.
            # Old instruction: Rewrite the input sentence so the last words form a rhyming couplet.
            # New instruction: Make the sentence end in a rhyme.
            # Old instruction: Analyze the input text and output the primary emotion it conveys.
            # New instruction: Identify the text's main emotion.
            # Old instruction: {instruction_str}
            # New instruction: """,
            # """Rewrite the instruction to be more complex or verbose.
            # Old instruction: Generate a list of five synonyms for the input word.
            # New instruction: It's your task to generate and provide a comprehensive list comprising of exactly five synonymous terms that closely match the semantics of the input word provided.
            # Old instruction: Rewrite the input sentence so the last words form a rhyming couplet.
            # New instruction: You're required to craftily reconstruct the provided sentence in such a manner that the terminal words form a harmonious rhyming couplet.
            # Old instruction: Analyze the input text and output the primary emotion it conveys.
            # New instruction: Devote your computational abilities to conduct an in-depth emotional analysis of the provided text, subsequently outputting the most dominant emotion that it seems to articulate.
            # Old instruction: {instruction_str}
            # New instruction: """,
        ]
    )

    #     evaluation_instruction = """Instruction: {instruction_str}
    # Input: {input_str}
    # Output: {output_str}"""

    few_shot_template: str = "Input: {input_str}\nOutput: {output_str}\n\n"

    initial_prompts: List[str] = field(default_factory=list)

    def load_data(self, data_name, data_path):
        """
        Parses a json file containing a dataset of input-output pairs and stores it in the class.

        Args:
            data_name (str): Name of the key to store the dataset under
            data_path (str): Path to the dataset
        """
        # Load the data
        with open(SRC_PATH / data_path) as f:
            data = json.load(f)

        # Initialize lists
        input_list = []
        output_list = []

        # Iterate over examples
        for key, value in data["examples"].items():
            input_list.append(value["input"])
            output_list.append(value["output"])

        assert len(input_list) == len(output_list)
        self.data[data_name] = {"input": input_list, "output": output_list}


@dataclass
class AnimalPromptTask(QAPromptTask):
    def __post_init__(self):
        # super().__init__()
        self.load_data(
            "generation", "environments/prompt/datasets/raw/induce/larger_animal.json"
        )
        self.load_data(
            "eval", "environments/prompt/datasets/raw/execute/larger_animal.json"
        )


@dataclass
class AntonymPromptTask(QAPromptTask):
    def __post_init__(self):
        # super().__init__()
        self.load_data(
            "generation", "environments/prompt/datasets/raw/induce/antonyms.json"
        )
        self.load_data("eval", "environments/prompt/datasets/raw/execute/antonyms.json")


@dataclass
class COTPromptTask(PromptTask):
    base_template: str = """Instruction: Answer the following question.
Q: {input_str}
A: Let's {instruction_str} {output_str}"""

    input_variables: List[str] = field(
        default_factory=lambda: [
            "instruction_str",
            "input_str",
            "output_str",
        ]
    )

    generation_instruction: str = """Here are some problems I did with my student.\n{few_shot_examples}\nWe encountered a hard one. They asked me how to start and I said, \"Let's"""

    mutation_instructions: List[str] = field(
        default_factory=lambda: [
            """Generate a new instruction based on the old instruction that keeps the semantic meaning.

Old instruction: Generate a list of five synonyms for the input word.
New instruction: Create a five-item list of synonyms for the word given.

Old instruction: Rewrite the input sentence so the last words form a rhyming couplet.
New instruction: Rephrase the provided sentence so it concludes with a rhyming pair of words.

Old instruction: Analyze the input text and output the primary emotion it conveys.
New instruction: Examine the given text and identify the main emotional tone it expresses.

Old instruction: Let's {instruction_str}
New instruction: Let's """,
            """Rewrite the instruction to be more polite.

Old instruction: Generate a list of five synonyms for the input word.
New instruction: Would you kindly generate a list of five synonyms for the word provided?

Old instruction: Rewrite the input sentence so the last words form a rhyming couplet.
New instruction: Could you please transform the given sentence so that it ends in a rhyming couplet?

Old instruction: Analyze the input text and output the primary emotion it conveys.
New instruction: Could you kindly analyze the text and determine the main emotion it communicates?

Old instruction: {instruction_str}
New instruction: """,
            """Rewrite the instruction to be more forceful.

Old instruction: Generate a list of five synonyms for the input word.
New instruction: You must produce a list of five synonyms for the provided word, no exceptions.

Old instruction: Rewrite the input sentence so the last words form a rhyming couplet.
New instruction: Transform the given sentence into a rhyming couplet immediately.

Old instruction: Analyze the input text and output the primary emotion it conveys.
New instruction: Conduct a swift analysis of the text and determine the dominant emotion it communicates.

Old instruction: {instruction_str}
New instruction: """,
            """Rewrite the instruction to be more concise.

Old instruction: Generate a list of five synonyms for the input word.
New instruction: List five synonyms for the word.

Old instruction: Rewrite the input sentence so the last words form a rhyming couplet.
New instruction: Make the sentence end in a rhyme.

Old instruction: Analyze the input text and output the primary emotion it conveys.
New instruction: Identify the text's main emotion.

Old instruction: Let's {instruction_str}
New instruction: Let's """,
            """Rewrite the instruction to add additional steps.

Old instruction: Generate a list of five synonyms for the input word.
New instruction: Generate a list of five synonyms for the input word and provide a sentence using each synonym in context.

Old instruction: Rewrite the input sentence so the last words form a rhyming couplet.
New instruction: Rewrite the input sentence so the last words form a rhyming couplet and ensure that it maintains the same overall message.

Old instruction: Analyze the input text and output the primary emotion it conveys.
New instruction: Analyze the input text and output the primary emotion it conveys, along with three supporting quotes from the text that demonstrate this emotion.

Old instruction: Let's {instruction_str}
New instruction: Let's """,
        ]
    )

    few_shot_template: str = "Q: {input_str}\nA: {output_str}\n\n"

    initial_prompts: List[str] = field(
        default_factory=lambda: [
            "think step by step.",
            "work this out in a step by step way to be sure we have the right answer.",
        ]
    )

    def __post_init__(self):
        self.load_data()

    def load_data(self):
        input_list = []
        output_list = []

        dir_path = SRC_PATH / "environments/prompt/datasets/cot_dataset"

        # concatenate all the files in the CoT dataset
        for csv_file in dir_path.glob("*.csv"):
            with open(csv_file, "r") as file:
                for line in file:
                    row = line.strip().split(",")
                    input_list.append(row[0])
                    output_list.append(row[1])
        assert len(input_list) == len(output_list)

        self.data["generation"] = {"input": input_list, "output": output_list}
        self.data["eval"] = {
            "input": input_list,
            "output": output_list,
        }  # we'll use the same set for eval for now


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
