"""Tests for the visualization module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.core.visualization import GraphVisualizer
from src.models.graph_models import Node, Relationship, GraphDocument
from src.exceptions.custom_exceptions import VisualizationError


class TestGraphVisualizer:
    """Test cases for GraphVisualizer."""
    
    @patch('src.core.visualization.Network')
    def test_create_network(self, mock_network_class):
        """Test network creation with default settings."""
        mock_network = MagicMock()
        mock_network_class.return_value = mock_network
        
        visualizer = GraphVisualizer()
        network = visualizer.create_network()
        
        mock_network_class.assert_called_once()
        assert network == mock_network
        mock_network.set_options.assert_called_once()
    
    @patch('src.core.visualization.Network')
    def test_visualize_simple_graph(self, mock_network_class):
        """Test visualization of a simple graph."""
        mock_network = MagicMock()
        mock_network_class.return_value = mock_network
        
        # Create simple graph
        node1 = Node(id="Alice", type="Person")
        node2 = Node(id="Bob", type="Person")
        rel = Relationship(source=node1, target=node2, type="KNOWS")
        
        doc = GraphDocument(nodes=[node1, node2], relationships=[rel])
        
        visualizer = GraphVisualizer()
        result = visualizer.visualize(doc, output_file="test.html")
        
        assert result == mock_network
        mock_network.add_node.assert_any_call("Alice", label="Alice", title="Type: Person", group="Person")
        mock_network.add_node.assert_any_call("Bob", label="Bob", title="Type: Person", group="Person")
        mock_network.add_edge.assert_called_once_with("Alice", "Bob", label="knows", title="Type: KNOWS")
        mock_network.save_graph.assert_called_once_with("test.html")
    
    @patch('src.core.visualization.Network')
    def test_visualize_with_properties(self, mock_network_class):
        """Test visualization with node and relationship properties."""
        mock_network = MagicMock()
        mock_network_class.return_value = mock_network
        
        # Create graph with properties
        node1 = Node(id="Alice", type="Person", properties={"age": "30"})
        node2 = Node(id="Company", type="Organization", properties={"founded": "2020"})
        rel = Relationship(source=node1, target=node2, type="WORKS_AT", properties={"since": "2021"})
        
        doc = GraphDocument(nodes=[node1, node2], relationships=[rel])
        
        visualizer = GraphVisualizer()
        visualizer.visualize(doc)
        
        # Check that properties are included in titles
        call_args = mock_network.add_node.call_args_list
        assert len(call_args) == 2
    
    @patch('src.core.visualization.Network')
    def test_get_graph_summary(self, mock_network_class):
        """Test graph summary generation."""
        node1 = Node(id="Alice", type="Person")
        node2 = Node(id="Bob", type="Person")
        node3 = Node(id="Company", type="Organization")
        node4 = Node(id="Charlie", type="Person")  # Isolated node
        
        rel1 = Relationship(source=node1, target=node2, type="KNOWS")
        rel2 = Relationship(source=node2, target=node3, type="WORKS_AT")
        
        doc = GraphDocument(nodes=[node1, node2, node3, node4], relationships=[rel1, rel2])
        
        visualizer = GraphVisualizer()
        summary = visualizer.get_graph_summary(doc)
        
        assert summary["total_nodes"] == 4
        assert summary["total_relationships"] == 2
        assert summary["connected_nodes"] == 3
        assert summary["isolated_nodes"] == 1
        assert summary["node_types"]["Person"] == 3
        assert summary["node_types"]["Organization"] == 1
    
    @patch('src.core.visualization.Network')
    def test_visualize_batch(self, mock_network_class):
        """Test batch visualization."""
        mock_network = MagicMock()
        mock_network_class.return_value = mock_network
        
        doc1 = GraphDocument(nodes=[], relationships=[], source="Text 1")
        doc2 = GraphDocument(nodes=[], relationships=[], source="Text 2")
        
        visualizer = GraphVisualizer()
        
        with patch('src.core.visualization.Path') as mock_path:
            mock_path.return_value.mkdir.return_value = None
            paths = visualizer.visualize_batch([doc1, doc2], output_dir="output", prefix="test")
        
        assert len(paths) == 2
        mock_network.save_graph.assert_called()
