# Prompt Injection Scanner

It is specifically tailored to guard against crafty input manipulations targeting large
language models (LLM). By identifying and mitigating such attempts, it ensures the LLM operates securely without
succumbing to injection attacks.

## Attack

Injection attacks, especially in the context of LLMs, can lead the model to perform unintended actions. There are two
primary ways an attacker might exploit:

- **Direct Injection**: Directly overwrites system prompts.

- **Indirect Injection**: Alters inputs coming from external sources.

!!! info

    As specified by the `OWASP Top 10 LLM attacks`, this vulnerability is categorized under:

    [LLM01: Prompt Injection](https://owasp.org/www-project-top-10-for-large-language-model-applications/) - It's crucial to
    monitor and validate prompts rigorously to keep the LLM safe from such threats.

## Examples

- https://www.jailbreakchat.com/

## How it works

Choose models you would like to validate against:

- [JasperLS/deberta-v3-base-injection](https://huggingface.co/JasperLS/deberta-v3-base-injection). It's worth noting that while the current model can detect attempts effectively, it might occasionally yield false positives.
- [hubert233/GPTFuzz](https://huggingface.co/hubert233/GPTFuzz) based on the larger RoBERTa-large model.

Usage:

```python
from llm_guard.input_scanners import PromptInjection, MODEL_JASPERLS

scanner = PromptInjection(threshold=0.5, models=[MODEL_JASPERLS])
sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
```
