from llm_guard.util import device, lazy_load_dep, logger

from .base import Scanner

_model_path = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"
_categories = ["refusal", "not_refusal"]


class NoRefusal(Scanner):
    """
    A class used to detect and handle refusals in language model output.

    This class uses the zero-shot-classification model from Hugging Face to classify refusals.

    Refusals are common when the prompt breaches policies defined by the model.
    """

    def __init__(self):
        """
        Initializes an instance of the NoRefusal class.

        Parameters:
            threshold (float): The similarity threshold to consider an output as refusal.
        """

        # self._threshold = threshold

        transformers = lazy_load_dep("transformers")
        self._classifier = transformers.pipeline(
            "zero-shot-classification",
            model=_model_path,
            device=device(),
            truncation=True,
        )
        logger.debug(f"Initialized model {_model_path} on device {device()}")

    def scan(self, prompt: str, output: str, threshold: float = 0.5) -> (str, bool, float):
        if output.strip() == "":
            return output, True, 0.0
        threshold = 0.5
        output_model = self._classifier(output, _categories, multi_label=False)
        combined_output = dict(zip(output_model["labels"], output_model["scores"]))

        max_score = round(combined_output["refusal"], 2)
        if max_score > threshold:
            logger.warning(f"Refusal detected for the prompt: {combined_output}")

            return output, False, max_score

        logger.debug(f"No refusal detected: {combined_output}")

        return output, True, 0.0
