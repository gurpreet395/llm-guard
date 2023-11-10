from typing import List

from llm_guard.util import device, lazy_load_dep, logger

from .base import Scanner

_model_path = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"


class BanTopics(Scanner):
    """
    BanTopics class is used to ban certain topics from the prompt.

    It uses a HuggingFace model to perform zero-shot classification.
    """

    def __init__(self, ):
        """
        Initialize BanTopics object.

        Args:
            topics (List[str]): List of topics to ban.
            threshold (float, optional): Threshold to determine if a topic is present in the prompt. Default is 0.75.

        Raises:
            ValueError: If no topics are provided.
        """

        transformers = lazy_load_dep("transformers")
        self._classifier = transformers.pipeline(
            "zero-shot-classification",
            model=_model_path,
            device=device(),
            truncation=True,
        )
        logger.debug(f"Initialized model {_model_path} on device {device()}")

    def scan(self, prompt: str, topics=List[str], threshold: float = 0.6) -> (str, bool, float):
        if prompt.strip() == "":
            return prompt, True, 0.0
        if len(topics) == 0:
            raise ValueError("No topics provided")

        self._threshold = threshold

        output_model = self._classifier(prompt, topics, multi_label=False)

        # max_score = round(max(output_model["scores"]) if output_model["scores"] else 0, 2)
        return prompt, True, output_model['scores'], output_model['labels']
        # if max_score > self._threshold:
        #     logger.warning(
        #         f"Topics detected for the prompt {output_model['labels']} with scores: {output_model['scores']}"
        #     )
        #
        #     return prompt, False, max_score
        #
        # logger.debug(
        #     f"No banned topics detected ({output_model['labels']}, scores: {output_model['scores']})"
        # )
        #
        # return prompt, True, 0.0
