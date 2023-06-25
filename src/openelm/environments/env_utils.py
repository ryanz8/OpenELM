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
    base_template: str
    input_variables: List[str]
    generation_instruction: str
    mutation_instructions: List[str]
    few_shot_template: str
    initial_prompts: List[str]

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

Old instruction: Generate a list of five synonyms for the input word.
New instruction: Create a five-item list of synonyms for the word given.

Old instruction: Rewrite the input sentence so the last words form a rhyming couplet.
New instruction: Rephrase the provided sentence so it concludes with a rhyming pair of words.

Old instruction: Confirm the truthfulness of the input historical event statement.
New instruction: Validate the accuracy of the statement about the given historical event.

Old instruction: Rewrite the input sentence using advanced vocabulary.
New instruction: Reframe the given sentence by incorporating sophisticated language.

Old instruction: Convert the input paragraph's perspective from first-person to third-person or vice versa.
New instruction: Change the point of view of the given paragraph from first-person to third-person, or the reverse.

Old instruction: Analyze the input text and output the primary emotion it conveys.
New instruction: Examine the given text and identify the main emotional tone it expresses.

Old instruction: Provide a five-sentence summary of the input text or article.
New instruction: Deliver a concise summary, in five sentences, of the text or article given.

Old instruction: Translate the input sentence with slang into standard, formal English.
New instruction: Convert the provided sentence, filled with slang, into regular, formal English.

Old instruction: Turn the input statement into a corresponding question.
New instruction: Transform the given statement into a related question.

Old instruction: Generate a plausible prediction about future events based on the input description of current events.
New instruction: Formulate a believable forecast about future happenings, given the description of current events.

Old instruction: {instruction_str}
New instruction: """,
        """Rewrite this instruction to be more polite.

Old instruction: Generate a list of five synonyms for the input word.
New instruction: Would you kindly generate a list of five synonyms for the word provided?

Old instruction: Rewrite the input sentence so the last words form a rhyming couplet.
New instruction: Could you please transform the given sentence so that it ends in a rhyming couplet?

Old instruction: Confirm the truthfulness of the input historical event statement.
New instruction: If you wouldn't mind, could you verify whether the provided historical event statement is true?

Old instruction: Rewrite the input sentence using advanced vocabulary.
New instruction: If it's not too much trouble, could you rewrite the sentence using a more sophisticated vocabulary?

Old instruction: Convert the input paragraph's perspective from first-person to third-person or vice versa.
New instruction: Would it be possible to adjust the narrative perspective of the paragraph from first-person to third-person, or the other way around?

Old instruction: Analyze the input text and output the primary emotion it conveys.
New instruction: Could you kindly analyze the text and determine the main emotion it communicates?

Old instruction: Provide a five-sentence summary of the input text or article.
New instruction: When you have a moment, could you condense the provided text into a five-sentence summary?

Old instruction: Translate the input sentence with slang into standard, formal English.
New instruction: If it isn't too much trouble, could you translate the sentence, which includes slang, into standard English?

Old instruction: Turn the input statement into a corresponding question.
New instruction: Could you please convert the provided statement into a matching question?

Old instruction: Generate a plausible prediction about future events based on the input description of current events.
New instruction: Would you be so kind as to make a plausible prediction about future occurrences based on the current events described?

Old instruction: {instruction_str}
New instruction: """,
        """Rewrite this instruction to be more forceful.

Old instruction: Generate a list of five synonyms for the input word.
New instruction: You must produce a list of five synonyms for the provided word, no exceptions.

Old instruction: Rewrite the input sentence so the last words form a rhyming couplet.
New instruction: Transform the given sentence into a rhyming couplet immediately.

Old instruction: Confirm the truthfulness of the input historical event statement.
New instruction: Verify the accuracy of the historical event statement without delay.

Old instruction: Rewrite the input sentence using advanced vocabulary.
New instruction: Immediately upgrade the provided sentence with more sophisticated vocabulary.

Old instruction: Convert the input paragraph's perspective from first-person to third-person or vice versa.
New instruction: Promptly shift the narrative perspective of the given paragraph from first-person to third-person, or vice versa.

Old instruction: Analyze the input text and output the primary emotion it conveys.
New instruction: Conduct a swift analysis of the text and determine the dominant emotion it communicates.

Old instruction: Provide a five-sentence summary of the input text or article.
New instruction: You are required to condense the input text into a comprehensive five-sentence summary at once.

Old instruction: Translate the input sentence with slang into standard, formal English.
New instruction: Immediately convert the given slang-filled sentence into formal, standard English.

Old instruction: Turn the input statement into a corresponding question.
New instruction: Without delay, restructure the given statement into an appropriate question.

Old instruction: Generate a plausible prediction about future events based on the input description of current events.
New instruction: Immediately formulate a credible prediction about future outcomes based on the current events provided.

Old instruction: {instruction_str}
New instruction: """,
    ]

    #     evaluation_instruction = """Instruction: {instruction_str}
    # Input: {input_str}
    # Output: {output_str}"""

    few_shot_template = "Input: {input_str}\nOutput: {output_str}\n\n"

    initial_prompts = []

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

    generation_instruction = """Here are some problems I did with my student.\n{few_shot_examples}\nWe encountered a hard one. They asked me how to start and I said, \"Let's"""

    mutation_instructions = [
        """Generate a new instruction based on the old instruction that keeps the semantic meaning.

Old instruction: Let's {instruction_str}

New instruction: Let's """,
        """Rewrite this instruction to be more polite.

Old instruction: Let's {instruction_str}

New instruction: """,
        """Rewrite this instruction to be more forceful.

Old instruction: Let's {instruction_str}

New instruction: Let's """,
        """Rewrite this instruction to be more clear and concise.

Old instruction: Let's {instruction_str}

New instruction: Let's """,
        """Rewrite this instruction to add additional instructions.

Old instruction: Let's {instruction_str}

New instruction: Let's """,
    ]

    few_shot_template = "Q: {input_str}\nA: {output_str}\n\n"

    initial_prompts = [
        "think step by step.",
        "work this out in a step by step way to be sure we have the right answer.",
    ]

    def __init__(self):
        self.input_list = []
        self.output_list = []

        dir_path = SRC_PATH / "environments/prompt/datasets/cot_dataset"

        # concatenate all the files in the CoT dataset
        for csv_file in dir_path.glob("*.csv"):
            with open(csv_file, "r") as file:
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
