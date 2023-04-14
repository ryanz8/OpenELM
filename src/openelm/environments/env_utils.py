import numpy as np
from torch import no_grad
from transformers import CLIPModel, CLIPProcessor


def get_image_target(name: str) -> np.ndarray:
    if name == "circle":
        target = np.zeros((32, 32, 3))
        for y in range(32):
            for x in range(32):
                if (y - 16) ** 2 + (x - 16) ** 2 <= 100:  # a radius-10 circle
                    target[y, x] = np.array([255, 255, 0])
    if name == "smiley":
        target = np.zeros((32, 32, 3))
        for y in range(32):
            for x in range(32):
                if (y - 16) ** 2 + (x - 16) ** 2 <= 100:  # a radius-10 circle
                    target[y, x] = np.array([255, 255, 0])

            # eyes
            target[13:16, 16 - 4] = np.array([0, 0, 0])
            target[13:16, 16 + 4] = np.array([0, 0, 0])

            # mouth
            target[20, 14:19] = np.array([0, 0, 0])
            target[19, 13] = np.array([0, 0, 0])
            target[19, 19] = np.array([0, 0, 0])
    else:
        raise NotImplementedError(f"Image target {name} not implemented")
    return target


class CLIPWrapper:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # zero out temperature to get unscaled logits (should just be cosine similarity)
        self.model.logit_scale.mul_(0.0)

    @no_grad()
    def __call__(self, image: np.ndarray, prompts: list[str]) -> np.ndarray:
        inputs = self.processor(
            images=image, text=prompts, return_tensors="pt", padding=True
        )
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # image-text similarity score
        # probs = logits_per_image.softmax(dim=1)  # label probabilities

        return logits_per_image.cpu().numpy()[0]


IMAGE_SEED: str = """
def draw():
\tpic = np.zeros((32, 32, 3))
\tfor x in range(2, 30):
\t\tfor y in range(2, 30):
\t\t\tpic[x, y] = np.array([0, 0, 255])
\treturn pic
"""

NULL_SEED: str = ""
