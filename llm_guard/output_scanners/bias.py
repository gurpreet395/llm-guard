from llm_guard.util import device, lazy_load_dep, logger

from .base import Scanner

_model_path = "valurank/distilroberta-bias"


class Bias(Scanner):
    """
    This class is designed to detect and evaluate potential biases in text using a pretrained model from HuggingFace.
    """

    def __init__(self):
        """
        Initializes the Bias scanner with a probability threshold for bias detection.

        Parameters:
           threshold (float): The threshold above which a text is considered biased.
                              Default is 0.75.
        """
        transformers = lazy_load_dep("transformers")
        self._classifier = transformers.pipeline(
            "text-classification",
            model=_model_path,
            device=device(),
            truncation=True,
        )
        logger.debug(f"Initialized model {_model_path} on device {device()}")

    def scan(self, prompt: str, output: str, threshold: float = 0.75) -> (str, bool, float):
        if output.strip() == "":
            return output, True, 0.0

        classifier_output = self._classifier(output)
        score = round(
            classifier_output[0]["score"]
            if classifier_output[0]["label"] == "BIASED"
            else 1 - classifier_output[0]["score"],
            2,
        )
        if score > threshold:
            logger.warning(
                f"Detected biased text with score: {score}, threshold: {threshold}"
            )

            return output, False, score

        logger.debug(f"Not biased result. Max score: {score}, threshold: {threshold}")

        return output, True, 0.0
