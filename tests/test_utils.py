"""Tests for the utility functions."""

import pytest
import json
from pathlib import Path
from unittest.mock import mock_open, patch

from src.utils.helpers import (
    chunk_text,
    sanitize_filename,
    format_graph_for_display,
    merge_graph_documents,
    save_graph_to_json,
    load_graph_from_json,
)
from src.models.graph_models import Node, Relationship, GraphDocument


class TestChunkText:
    """Test cases for chunk_text function."""
    
    def test_short_text_no_chunking(self):
        """Test that short text is not chunked."""
        text = "Short text."
        result = chunk_text(text, chunk_size=1000)
        assert result == [text]
    
    def test_long_text_chunking(self):
        """Test that long text is chunked correctly."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        result = chunk_text(text, chunk_size=30, chunk_overlap=5)
        
        assert len(result) > 1
        # Each chunk should end with a sentence or word boundary
        for chunk in result:
            assert len(chunk) <= 30 or chunk.endswith(('.', '!', '?', ' '))
    
    def test_chunking_respects_sentence_boundary(self):
        """Test that chunking respects sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_text(text, chunk_size=25, chunk_overlap=5)
        
        # Should break at sentence boundaries when possible
        for chunk in result:
            assert chunk.strip().endswith('.') or len(chunk.strip()) < 25


class TestSanitizeFilename:
    """Test cases for sanitize_filename function."""
    
    def test_removes_invalid_chars(self):
        """Test removal of invalid filename characters."""
        filename = 'file<name>:"/\\|?*.txt'
        result = sanitize_filename(filename)
        assert '<' not in result
        assert ':' not in result
        assert '/' not in result
        assert '|' not in result
    
    def test_handles_long_names(self):
        """Test that long names are truncated."""
        filename = 'a' * 300
        result = sanitize_filename(filename)
        assert len(result) <= 200
    
    def test_handles_empty_name(self):
        """Test handling of empty or whitespace-only names."""
        assert sanitize_filename('   ') == 'unnamed'
        assert sanitize_filename('...') == 'unnamed'


class TestGraphOperations:
    """Test cases for graph operations."""
    
    def test_merge_graph_documents(self):
        """Test merging multiple graph documents."""
        node1 = Node(id="Alice", type="Person")
        node2 = Node(id="Bob", type="Person")
        node3 = Node(id="Charlie", type="Person")
        
        rel1 = Relationship(source=node1, target=node2, type="KNOWS")
        rel2 = Relationship(source=node2, target=node3, type="KNOWS")
        
        doc1 = GraphDocument(nodes=[node1, node2], relationships=[rel1])
        doc2 = GraphDocument(nodes=[node2, node3], relationships=[rel2])
        
        merged = merge_graph_documents([doc1, doc2])
        
        assert len(merged.nodes) == 3
        assert len(merged.relationships) == 2
    
    def test_format_graph_for_display(self):
        """Test graph display formatting."""
        node1 = Node(id="Alice", type="Person", properties={"age": "30"})
        node2 = Node(id="Bob", type="Person")
        rel = Relationship(source=node1, target=node2, type="KNOWS")
        
        doc = GraphDocument(nodes=[node1, node2], relationships=[rel])
        
        result = format_graph_for_display(doc)
        
        assert "KNOWLEDGE GRAPH" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "KNOWS" in result
        assert "Person" in result


class TestJsonOperations:
    """Test cases for JSON save/load operations."""
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_graph_to_json(self, mock_json_dump, mock_file):
        """Test saving graph to JSON."""
        node = Node(id="Alice", type="Person")
        doc = GraphDocument(nodes=[node], relationships=[])
        
        with patch('pathlib.Path.mkdir'):
            result = save_graph_to_json(doc, "test.json")
        
        assert str(result) == "test.json"
        mock_json_dump.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({
        "nodes": [{"id": "Alice", "type": "Person", "properties": {}}],
        "relationships": [],
        "source": "test"
    }))
    def test_load_graph_from_json(self, mock_file):
        """Test loading graph from JSON."""
        result = load_graph_from_json("test.json")
        
        assert len(result.nodes) == 1
        assert result.nodes[0].id == "Alice"
        assert result.nodes[0].type == "Person"
        assert result.source == "test"
