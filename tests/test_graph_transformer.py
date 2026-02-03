"""Tests for the graph transformer module."""

import pytest
from unittest.mock import Mock, patch

from src.core.graph_transformer import GraphTransformer
from src.models.graph_models import GraphSchema, Node, Relationship, GraphDocument
from src.exceptions.custom_exceptions import ConfigurationError, ExtractionError


class TestGraphTransformer:
    """Test cases for GraphTransformer."""
    
    @patch('src.core.graph_transformer.ChatOpenAI')
    @patch('src.core.graph_transformer.settings')
    def test_init_without_api_key(self, mock_settings, mock_chat_openai):
        """Test that initialization fails without API key."""
        mock_settings.OPENAI_API_KEY = None
        
        with pytest.raises(ConfigurationError):
            GraphTransformer()
    
    @patch('src.core.graph_transformer.ChatOpenAI')
    @patch('src.core.graph_transformer.settings')
    def test_init_with_api_key(self, mock_settings, mock_chat_openai):
        """Test successful initialization with API key."""
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.DEFAULT_LLM_MODEL = "gpt-4.1-mini"
        mock_settings.DEFAULT_TEMPERATURE = 0.0
        
        transformer = GraphTransformer()
        
        assert transformer.model_name == "gpt-4.1-mini"
        assert transformer.temperature == 0.0
        assert transformer.llm is not None
    
    @patch('src.core.graph_transformer.ChatOpenAI')
    @patch('src.core.graph_transformer.settings')
    def test_init_with_custom_params(self, mock_settings, mock_chat_openai):
        """Test initialization with custom parameters."""
        mock_settings.OPENAI_API_KEY = "test-key"
        
        schema = GraphSchema(
            allowed_nodes=["Person", "Organization"],
            allowed_relationships=[("Person", "WORKS_AT", "Organization")],
        )
        
        transformer = GraphTransformer(
            model="gpt-4",
            temperature=0.5,
            schema=schema,
        )
        
        assert transformer.model_name == "gpt-4"
        assert transformer.temperature == 0.5
        assert transformer.schema == schema
    
    def test_convert_to_graph_documents(self):
        """Test conversion of raw documents to GraphDocument format."""
        # Mock raw document structure
        mock_raw_doc = Mock()
        mock_raw_doc.nodes = []
        mock_raw_doc.relationships = []
        mock_raw_doc.source = Mock()
        mock_raw_doc.source.page_content = "Test content"
        
        transformer = Mock(spec=GraphTransformer)
        transformer._convert_to_graph_documents = GraphTransformer._convert_to_graph_documents
        
        result = transformer._convert_to_graph_documents(transformer, [mock_raw_doc])
        
        assert len(result) == 1
        assert result[0].source == "Test content"
    
    def test_get_stats(self):
        """Test statistics calculation."""
        # Create test graph documents
        node1 = Node(id="Alice", type="Person")
        node2 = Node(id="Bob", type="Person")
        node3 = Node(id="Company", type="Organization")
        
        rel1 = Relationship(source=node1, target=node3, type="WORKS_AT")
        rel2 = Relationship(source=node2, target=node3, type="WORKS_AT")
        
        doc1 = GraphDocument(nodes=[node1, node2], relationships=[rel1], source="Text 1")
        doc2 = GraphDocument(nodes=[node3], relationships=[rel2], source="Text 2")
        
        transformer = Mock(spec=GraphTransformer)
        transformer.get_stats = GraphTransformer.get_stats
        
        stats = transformer.get_stats(transformer, [doc1, doc2])
        
        assert stats["num_documents"] == 2
        assert stats["total_nodes"] == 3
        assert stats["total_relationships"] == 2
        assert stats["unique_node_types"] == 2
        assert stats["unique_relationship_types"] == 1


class TestGraphSchema:
    """Test cases for GraphSchema."""
    
    def test_to_llm_transformer_kwargs(self):
        """Test conversion to LLMGraphTransformer kwargs."""
        schema = GraphSchema(
            allowed_nodes=["Person", "Organization"],
            allowed_relationships=[("Person", "WORKS_AT", "Organization")],
            node_properties=["name", "age"],
            relationship_properties=["since"],
            strict_mode=True,
        )
        
        kwargs = schema.to_llm_transformer_kwargs()
        
        assert kwargs["allowed_nodes"] == ["Person", "Organization"]
        assert kwargs["allowed_relationships"] == [("Person", "WORKS_AT", "Organization")]
        assert kwargs["node_properties"] == ["name", "age"]
        assert kwargs["relationship_properties"] == ["since"]
        assert kwargs["strict_mode"] == True
    
    def test_to_llm_transformer_kwargs_empty(self):
        """Test conversion with empty schema."""
        schema = GraphSchema()
        
        kwargs = schema.to_llm_transformer_kwargs()
        
        assert "allowed_nodes" not in kwargs
        assert "allowed_relationships" not in kwargs
        assert "strict_mode" in kwargs
