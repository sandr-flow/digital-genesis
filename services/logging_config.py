"""Centralized logging configuration."""

import logging
import os
from config import LOG_DIR


def setup_logging():
    """Configure all system loggers.

    Loggers:
        - ThoughtProcess: thinking and dialogue process
        - Reflections: reflection results
        - Concepts: cognitive asset extraction
    """
    # Create logs directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # --- Thought process logger ---
    thought_process_logger = logging.getLogger("ThoughtProcess")
    thought_process_logger.setLevel(logging.INFO)
    thought_process_logger.propagate = False
    
    thought_handler = logging.StreamHandler()
    thought_handler.setFormatter(
        logging.Formatter('%(asctime)s - [ThoughtProcess] - %(message)s')
    )
    thought_process_logger.addHandler(thought_handler)
    
    # --- Reflections logger ---
    reflections_logger = logging.getLogger("Reflections")
    reflections_logger.setLevel(logging.INFO)
    reflections_logger.propagate = False
    
    reflections_file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "reflections.log"), 
        encoding='utf-8'
    )
    reflections_file_handler.setFormatter(
        logging.Formatter('%(asctime)s\\n%(message)s\\n' + '-' * 80)
    )
    reflections_logger.addHandler(reflections_file_handler)
    
    # --- Concepts logger ---
    concepts_logger = logging.getLogger("Concepts")
    concepts_logger.setLevel(logging.INFO)
    concepts_logger.propagate = False
    
    concepts_handler = logging.StreamHandler()
    concepts_handler.setFormatter(
        logging.Formatter('%(asctime)s - [CONCEPTS] - %(message)s')
    )
    concepts_logger.addHandler(concepts_handler)
    
    logging.info("Logging system initialized")
    
    return thought_process_logger, reflections_logger, concepts_logger


def get_thought_logger():
    """Return the thought process logger."""
    return logging.getLogger("ThoughtProcess")


def get_reflections_logger():
    """Return the reflections logger."""
    return logging.getLogger("Reflections")


def get_concepts_logger():
    """Return the concepts logger."""
    return logging.getLogger("Concepts")
