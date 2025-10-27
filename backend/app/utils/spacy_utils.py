import logging
import os
from time import sleep
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

_SPACY_AVAILABLE = False
_SPACY_MODELS: Dict[str, str] = {}
_SPACY_LOADED_MODELS: Dict[str, 'spacy.Language'] = {}
_SPACY_LOAD_RETRIES = max(0, int(os.getenv("SPACY_LOAD_RETRIES", "0")))
_SPACY_LOAD_RETRY_DELAY_SEC = max(0.0, float(os.getenv("SPACY_LOAD_RETRY_DELAY_SEC", "0.10")))

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
                if attempt < attempts:
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
    """Legacy function - kept for backward compatibility."""
    if not _SPACY_AVAILABLE:
        return False
    if language not in _SPACY_MODELS:
        return False
    return text_length > 249


def validate_text_splitting_configuration(text_length: int, language: str, enable_splitting: bool) -> None:
    """Legacy function - kept for backward compatibility."""
    if text_length > 249 and not enable_splitting:
        if not _SPACY_AVAILABLE:
            logger.warning(f"Text is {text_length} chars but spaCy not available.")
        elif language not in _SPACY_MODELS:
            logger.warning(f"Text is {text_length} chars but no spaCy model for '{language}'.")
    logger.info(
        f"Text splitting: {'enabled' if enable_splitting else 'disabled'} "
        f"(len={text_length}, spacy_model={_SPACY_MODELS.get(language, 'none')})"
    )


def get_spacy_model(language: str) -> Optional['spacy.Language']:
    """Get loaded spaCy model for language, loading it if necessary."""
    if not _SPACY_AVAILABLE or language not in _SPACY_MODELS:
        return None
    
    # Return cached model if available
    if language in _SPACY_LOADED_MODELS:
        return _SPACY_LOADED_MODELS[language]
    
    # Load model
    model_name = _SPACY_MODELS[language]
    try:
        model = spacy.load(model_name)
        _SPACY_LOADED_MODELS[language] = model
        logger.info(f"Loaded spaCy model '{model_name}' for language '{language}'")
        return model
    except Exception as e:
        logger.error(f"Failed to load spaCy model '{model_name}' for language '{language}': {e}")
        return None


def split_text_by_sentences(text: str, language: str, max_length: int = 250) -> List[str]:
    """
    Split text into sentences using spaCy, ensuring no sentences are lost.
    
    Args:
        text: Input text to split
        language: Language code
        max_length: Maximum length per segment
        
    Returns:
        List of text segments, each containing complete sentences
    """
    if not _SPACY_AVAILABLE or language not in _SPACY_MODELS:
        # Fallback: simple character-based splitting
        logger.warning(f"No spaCy model for '{language}', using character-based splitting")
        segments = []
        for i in range(0, len(text), max_length):
            segments.append(text[i:i + max_length])
        return segments
    
    model = get_spacy_model(language)
    if not model:
        # Fallback: simple character-based splitting
        logger.warning(f"Failed to load spaCy model for '{language}', using character-based splitting")
        segments = []
        for i in range(0, len(text), max_length):
            segments.append(text[i:i + max_length])
        return segments
    
    try:
        doc = model(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not sentences:
            return [text] if text.strip() else []
        
        # Group sentences into segments that don't exceed max_length
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max_length, start a new segment
            if current_segment and len(current_segment) + len(sentence) + 1 > max_length:
                segments.append(current_segment.strip())
                current_segment = sentence
            else:
                if current_segment:
                    current_segment += " " + sentence
                else:
                    current_segment = sentence
        
        # Add the last segment if it's not empty
        if current_segment.strip():
            segments.append(current_segment.strip())
        
        logger.info(f"Split text into {len(segments)} segments using spaCy sentence boundaries")
        return segments
        
    except Exception as e:
        logger.error(f"spaCy sentence segmentation failed for language '{language}': {e}")
        # Fallback: simple character-based splitting
        segments = []
        for i in range(0, len(text), max_length):
            segments.append(text[i:i + max_length])
        return segments


def validate_sentence_boundaries(text: str, language: str) -> Dict[str, Any]:
    """
    Validate that spaCy sentence boundary detection is working correctly.
    
    Returns:
        Dictionary with validation results and statistics
    """
    if not _SPACY_AVAILABLE or language not in _SPACY_MODELS:
        return {
            "available": False,
            "reason": "spaCy not available or no model for language",
            "sentence_count": 0,
            "segments": []
        }
    
    model = get_spacy_model(language)
    if not model:
        return {
            "available": False,
            "reason": "Failed to load spaCy model",
            "sentence_count": 0,
            "segments": []
        }
    
    try:
        doc = model(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        segments = split_text_by_sentences(text, language)
        
        return {
            "available": True,
            "sentence_count": len(sentences),
            "segment_count": len(segments),
            "sentences": sentences,
            "segments": segments,
            "total_chars": len(text),
            "avg_sentence_length": sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        }
    except Exception as e:
        return {
            "available": False,
            "reason": f"spaCy processing failed: {e}",
            "sentence_count": 0,
            "segments": []
        }


