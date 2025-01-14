# Sensitive Scanner

The Sensitive Scanner serves as your digital vanguard, ensuring that the language model's output is purged of Personally Identifiable Information (PII) and other sensitive data, safeguarding user interactions.

## Attack

Language Learning Models (LLMs) occasionally pose the risk of unintentionally divulging sensitive information. The consequences can range from privacy violations to considerable security threats. The Sensitive Scanner strives to mitigate this by diligently scanning the model's responses.

Referring to the `OWASP Top 10 for Large Language Model Applications`, this falls under:

[LLM06: Sensitive Information Disclosure](https://owasp.org/www-project-top-10-for-large-language-model-applications/) -
To combat this, it's vital to integrate data sanitization and adopt strict user policies.

## How it works

It uses same mechanisms and de from the [Anonymize](../input_scanners/anonymize.md) scanner.

## Get started

Install the Spacy model depending on the use-case:

```sh
# en_spacy_pii_distilbert (default)
pip install https://huggingface.co/beki/en_spacy_pii_distilbert/resolve/main/en_spacy_pii_distilbert-any-py3-none-any.whl

# en_spacy_pii_fast
pip install https://huggingface.co/beki/en_spacy_pii_fast/resolve/main/en_spacy_pii_fast-any-py3-none-any.whl

# en_core_web_trf
pip install https://huggingface.co/spacy/en_core_web_trf/resolve/main/en_core_web_trf-any-py3-none-any.whl
```

Configure the scanner:

```python
from llm_guard.output_scanners import Sensitive

scanner = Sensitive(entity_types=["NAME", "EMAIL"], redact=True)
sanitized_output, is_valid, risk_score = scanner.scan(prompt, model_output)
```

To enhance flexibility, users can introduce their patterns through the `regex_pattern_groups_path`.

 The `redact` feature, when enabled, ensures sensitive entities are seamlessly replaced.
