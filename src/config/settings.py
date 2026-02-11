"""Configuration settings for the Knowledge Graph Generator."""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings and configuration."""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    OUTPUT_DIR = BASE_DIR / "output"
    DATA_DIR = BASE_DIR / "data"
    
    # Ensure directories exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    
    # LLM Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4.1-mini")
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0"))
    
    # Graph Configuration
    DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
    DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))
    
    # Graph Schema (Optional - for consistent extractions)
    # Example: ["Person", "Organization", "Location", "Award"]
    ALLOWED_NODES: Optional[List[str]] = None
    
    # Example: [("Person", "WORKS_AT", "Organization"), ...]
    ALLOWED_RELATIONSHIPS: Optional[List[tuple]] = None
    
    # Node and Relationship Properties
    # Example: ["birth_date", "death_date", "start_date"]
    NODE_PROPERTIES: Optional[List[str]] = None
    RELATIONSHIP_PROPERTIES: Optional[List[str]] = None
    
    # Extraction Settings
    STRICT_MODE = os.getenv("STRICT_MODE", "true").lower() == "true"
    IGNORE_TOOL_USAGE = os.getenv("IGNORE_TOOL_USAGE", "false").lower() == "true"
    
    # Neo4j Database Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    NEO4J_TIMEOUT = float(os.getenv("NEO4J_TIMEOUT", "30"))
    
    # Visualization Settings
    DEFAULT_GRAPH_HEIGHT = "1200px"
    DEFAULT_GRAPH_WIDTH = "100%"
    DEFAULT_GRAPH_BG_COLOR = "#222222"
    DEFAULT_GRAPH_FONT_COLOR = "white"
    
    # Physics Settings for PyVis
    PHYSICS_GRAVITATIONAL_CONSTANT = -100
    PHYSICS_CENTRAL_GRAVITY = 0.01
    PHYSICS_SPRING_LENGTH = 200
    PHYSICS_SPRING_CONSTANT = 0.08
    PHYSICS_MIN_VELOCITY = 0.75
    
    @classmethod
    def validate(cls) -> None:
        """Validate that required settings are configured."""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file "
                "or as an environment variable."
            )
    
    @classmethod
    def get_llm_config(cls) -> dict:
        """Get LLM configuration as a dictionary."""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "model": cls.DEFAULT_LLM_MODEL,
            "temperature": cls.DEFAULT_TEMPERATURE,
        }
    
    @classmethod
    def get_chunk_config(cls) -> dict:
        """Get chunking configuration as a dictionary.
        
        Note: These are defaults. Use sidebar values in UI for runtime configuration.
        """
        return {
            "chunk_size": cls.DEFAULT_CHUNK_SIZE,
            "chunk_overlap": cls.DEFAULT_CHUNK_OVERLAP,
        }
    
    @classmethod
    def get_neo4j_config(cls) -> dict:
        """Get Neo4j configuration as a dictionary."""
        return {
            "url": cls.NEO4J_URI,
            "username": cls.NEO4J_USERNAME,
            "password": cls.NEO4J_PASSWORD,
            "database": cls.NEO4J_DATABASE,
            "timeout": cls.NEO4J_TIMEOUT,
        }
    
    @classmethod
    def validate_neo4j(cls) -> None:
        """Validate that required Neo4j settings are configured."""
        if not cls.NEO4J_PASSWORD:
            raise ValueError(
                "NEO4J_PASSWORD not found. Please set it in your .env file "
                "or as an environment variable."
            )


# Global settings instance
settings = Settings()
