"""Centralized logging configuration for the Knowledge Graph Generator."""
import logging
import sys
from pathlib import Path

# Get project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Default log file location - logs/app.log in project root
DEFAULT_LOG_FILE = PROJECT_ROOT / 'logs' / 'app.log'


def setup_logger(name: str = "graph_knowledge", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Setup centralized logging for the application.
    
    Args:
        name: Logger name (default: "graph_knowledge")
        log_file: Optional log file path. If None, only console logging is enabled.
                 Defaults to DEFAULT_LOG_FILE if not specified.
        level: Logging level (default: logging.INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler - INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional) - DEBUG and above with more details
    if log_file is not False:  # False means explicitly disable file logging
        log_path = Path(log_file) if log_file else DEFAULT_LOG_FILE
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get an existing logger or create a new one with default settings.
    
    Args:
        name: Logger name. If None, returns the root 'graph_knowledge' logger.
    
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"graph_knowledge.{name}")
    return logging.getLogger("graph_knowledge")


# Pre-configured loggers for key components
# Used for graph extraction and transformation operations
logger_transformer = get_logger("transformer")

# Used for graph-based RAG queries and context retrieval
logger_rag = get_logger("rag")

# Used for graph visualization and HTML generation
logger_visualizer = get_logger("visualizer")

# Used for main application lifecycle and orchestration
logger_app = get_logger("app")

# Used for utility functions and helpers
logger_utils = get_logger("utils")

# Used for configuration and settings
logger_config = get_logger("config")
