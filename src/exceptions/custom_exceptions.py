"""Custom exceptions for the Knowledge Graph Generator."""


class KnowledgeGraphError(Exception):
    """Base exception for the Knowledge Graph Generator."""
    pass


class ConfigurationError(KnowledgeGraphError):
    """Raised when there's a configuration error."""
    pass


class ExtractionError(KnowledgeGraphError):
    """Raised when graph extraction fails."""
    pass


class ValidationError(KnowledgeGraphError):
    """Raised when graph validation fails."""
    pass


class VisualizationError(KnowledgeGraphError):
    """Raised when graph visualization fails."""
    pass


class LLMError(KnowledgeGraphError):
    """Raised when LLM processing fails."""
    pass
