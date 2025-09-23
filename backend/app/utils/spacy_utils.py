import logging
import os
from time import sleep
from typing import Dict

logger = logging.getLogger(__name__)

_SPACY_AVAILABLE = False
_SPACY_MODELS: Dict[str, str] = {}
_SPACY_LOAD_RETRIES = max(0, int(os.getenv("SPACY_LOAD_RETRIES", "1")))
_SPACY_LOAD_RETRY_DELAY_SEC = max(0.0, float(os.getenv("SPACY_LOAD_RETRY_DELAY_SEC", "0.25")))

try:
    import spacy
    _SPACY_AVAILABLE = True
    language_models = {
        "en": ["en_core_web_md", "en_core_web_sm"],
        "es": ["es_core_news_sm", "xx_ent_wiki_sm"],
        "fr": ["fr_core_news_sm", "xx_ent_wiki_sm"],
        "de": ["de_core_news_sm", "xx_ent_wiki_sm"],
        "it": ["it_core_news_sm", "xx_ent_wiki_sm"],
        "pt": ["pt_core_news_sm", "xx_ent_wiki_sm"],
        "pl": ["pl_core_news_sm", "xx_ent_wiki_sm"],
        "ru": ["ru_core_news_sm", "xx_ent_wiki_sm"],
        "nl": ["nl_core_news_sm", "xx_ent_wiki_sm"],
        "zh-cn": ["zh_core_web_sm", "xx_ent_wiki_sm"],
        "ja": ["ja_core_news_sm", "xx_ent_wiki_sm"],
        "tr": ["xx_ent_wiki_sm"],
        "cs": ["xx_ent_wiki_sm"],
        "ar": ["xx_ent_wiki_sm"],
        "hu": ["xx_ent_wiki_sm"],
        "ko": ["xx_ent_wiki_sm"]
    }
    def _try_load(model_name: str) -> bool:
        attempts = 1 + _SPACY_LOAD_RETRIES
        for attempt in range(attempts):
            try:
                spacy.load(model_name)
                return True
            except OSError as e:
                if attempt < attempts - 1:
                    logger.warning(f"spaCy load failed for '{model_name}' (attempt {attempt+1}/{attempts}): {e}")
                    if _SPACY_LOAD_RETRY_DELAY_SEC:
                        sleep(_SPACY_LOAD_RETRY_DELAY_SEC)
                    continue
                return False

    for lang, models in language_models.items():
        for m in models:
            if _try_load(m):
                _SPACY_MODELS[lang] = m
                break
    if not _SPACY_MODELS:
        _SPACY_AVAILABLE = False
        logger.warning("No spaCy models found, text splitting disabled")
    else:
        logger.info(f"spaCy models: {_SPACY_MODELS}")
except ImportError:
    _SPACY_AVAILABLE = False
    logger.warning("spaCy not available, text splitting disabled")


def should_enable_text_splitting(text_length: int, language: str) -> bool:
    if not _SPACY_AVAILABLE:
        return False
    if language not in _SPACY_MODELS:
        return False
    return text_length > 1000


def validate_text_splitting_configuration(text_length: int, language: str, enable_splitting: bool) -> None:
    if text_length > 1000 and not enable_splitting:
        if not _SPACY_AVAILABLE:
            logger.warning(f"Text is {text_length} chars but spaCy not available.")
        elif language not in _SPACY_MODELS:
            logger.warning(f"Text is {text_length} chars but no spaCy model for '{language}'.")
    logger.info(
        f"Text splitting: {'enabled' if enable_splitting else 'disabled'} "
        f"(len={text_length}, spacy_model={_SPACY_MODELS.get(language, 'none')})"
    )


