"""Graph visualization using PyVis."""

from pathlib import Path
from typing import List, Optional, Union

from pyvis.network import Network

from src.config.settings import settings
from src.config.logging_config import logger_visualizer
from src.models.graph_models import GraphDocument, Node, Relationship
from src.exceptions.custom_exceptions import VisualizationError


class GraphVisualizer:
    """
    A class for visualizing knowledge graphs using PyVis.
    
    This class provides methods to create interactive HTML visualizations
    of knowledge graphs extracted from text.
    
    Features:
    - Interactive HTML visualization with PyVis
    - Configurable physics and layout options
    - Node grouping by type
    - Edge filtering
    
    Example:
        >>> visualizer = GraphVisualizer()
        >>> graph = await transformer.extract_graph(text)
        >>> visualizer.visualize(graph[0], output_file="graph.html")
    """
    
    def __init__(
        self,
        height: Optional[str] = None,
        width: Optional[str] = None,
        bgcolor: Optional[str] = None,
        font_color: Optional[str] = None,
        directed: bool = True,
    ):
        """
        Initialize the GraphVisualizer.
        
        Args:
            height: Graph height (default from settings)
            width: Graph width (default from settings)
            bgcolor: Background color (default from settings)
            font_color: Font color (default from settings)
            directed: Whether the graph is directed
        """
        self.height = height or settings.DEFAULT_GRAPH_HEIGHT
        self.width = width or settings.DEFAULT_GRAPH_WIDTH
        self.bgcolor = bgcolor or settings.DEFAULT_GRAPH_BG_COLOR
        self.font_color = font_color or settings.DEFAULT_GRAPH_FONT_COLOR
        self.directed = directed
    
    def create_network(self) -> Network:
        """
        Create a new PyVis Network instance with default configuration.
        
        Returns:
            Configured Network instance
        """
        net = Network(
            height=self.height,
            width=self.width,
            directed=self.directed,
            notebook=False,
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            filter_menu=True,
            cdn_resources='remote',
        )
        
        # Configure physics
        net.set_options(f"""
        {{
            "physics": {{
                "forceAtlas2Based": {{
                    "gravitationalConstant": {settings.PHYSICS_GRAVITATIONAL_CONSTANT},
                    "centralGravity": {settings.PHYSICS_CENTRAL_GRAVITY},
                    "springLength": {settings.PHYSICS_SPRING_LENGTH},
                    "springConstant": {settings.PHYSICS_SPRING_CONSTANT}
                }},
                "minVelocity": {settings.PHYSICS_MIN_VELOCITY},
                "solver": "forceAtlas2Based"
            }}
        }}
        """)
        
        return net
    
    def visualize(
        self,
        graph_document: GraphDocument,
        output_file: Optional[Union[str, Path]] = None,
        filter_isolated_nodes: bool = True,
    ) -> Network:
        """
        Visualize a knowledge graph document.
        
        Args:
            graph_document: GraphDocument to visualize
            output_file: Path to save the HTML file (default: auto-generated)
            filter_isolated_nodes: Whether to filter out isolated nodes
        
        Returns:
            PyVis Network instance
        
        Raises:
            VisualizationError: If visualization fails
        """
        num_nodes = len(graph_document.nodes)
        num_rels = len(graph_document.relationships)
        logger_visualizer.info(f"Creating visualization: {num_nodes} nodes, {num_rels} relationships")
        
        try:
            # Create network
            net = self.create_network()
            
            # Get nodes and relationships
            nodes = graph_document.nodes
            relationships = graph_document.relationships
            
            # Build lookup for valid nodes
            node_dict = {node.id: node for node in nodes}
            
            # Filter and validate relationships
            valid_edges = []
            valid_node_ids = set()
            
            for rel in relationships:
                if rel.source.id in node_dict and rel.target.id in node_dict:
                    valid_edges.append(rel)
                    valid_node_ids.update([rel.source.id, rel.target.id])
            
            # If filtering isolated nodes, only include connected nodes
            if filter_isolated_nodes:
                nodes_to_add = valid_node_ids
                isolated_count = len(nodes) - len(valid_node_ids)
                if isolated_count > 0:
                    logger_visualizer.debug(f"Filtered out {isolated_count} isolated nodes")
            else:
                nodes_to_add = set(node_dict.keys())
            
            # Add nodes to the graph
            added_nodes = 0
            for node_id in nodes_to_add:
                node = node_dict[node_id]
                try:
                    # Create label with type information if available
                    title = f"Type: {node.type}"
                    if node.properties:
                        props_str = "\n".join([f"{k}: {v}" for k, v in node.properties.items()])
                        title += f"\n\nProperties:\n{props_str}"
                    
                    net.add_node(
                        node.id,
                        label=node.id,
                        title=title,
                        group=node.type,
                    )
                    added_nodes += 1
                except Exception as e:
                    # Log error but continue with other nodes
                    logger_visualizer.warning(f"Could not add node {node_id}: {e}")
                    continue
            
            # Add edges to the graph
            added_edges = 0
            for rel in valid_edges:
                try:
                    # Create label with type and properties
                    label = rel.type.lower()
                    title = f"Type: {rel.type}"
                    if rel.properties:
                        props_str = "\n".join([f"{k}: {v}" for k, v in rel.properties.items()])
                        title += f"\n\nProperties:\n{props_str}"
                    
                    net.add_edge(
                        rel.source.id,
                        rel.target.id,
                        label=label,
                        title=title,
                    )
                    added_edges += 1
                except Exception as e:
                    # Log error but continue with other edges
                    logger_visualizer.warning(f"Could not add edge {rel.source.id} -> {rel.target.id}: {e}")
                    continue
            
            logger_visualizer.debug(f"Added {added_nodes} nodes and {added_edges} edges to visualization")
            
            # Save the graph
            if output_file is None:
                output_file = settings.OUTPUT_DIR / "knowledge_graph.html"
            else:
                output_file = Path(output_file)
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                net.save_graph(str(output_file))
                logger_visualizer.info(f"Graph visualization saved to {output_file}")
            except Exception as e:
                logger_visualizer.error(f"Failed to save graph visualization: {str(e)}")
                raise VisualizationError(f"Failed to save graph: {str(e)}")
            
            return net
            
        except Exception as e:
            logger_visualizer.error(f"Graph visualization failed: {str(e)}", exc_info=True)
            raise VisualizationError(f"Visualization failed: {str(e)}")
    
    def visualize_batch(
        self,
        graph_documents: List[GraphDocument],
        output_dir: Optional[Union[str, Path]] = None,
        prefix: str = "graph",
    ) -> List[Path]:
        """
        Visualize multiple graph documents.
        
        Args:
            graph_documents: List of GraphDocument objects
            output_dir: Directory to save HTML files (default from settings)
            prefix: Prefix for output filenames
        
        Returns:
            List of paths to saved HTML files
        """
        if output_dir is None:
            output_dir = settings.OUTPUT_DIR
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for i, doc in enumerate(graph_documents):
            output_file = output_dir / f"{prefix}_{i+1:03d}.html"
            self.visualize(doc, output_file=output_file)
            saved_files.append(output_file)
        
        return saved_files
    
    def get_graph_summary(self, graph_document: GraphDocument) -> dict:
        """
        Get a summary of the graph for display purposes.
        
        Args:
            graph_document: GraphDocument to summarize
        
        Returns:
            Dictionary containing summary information
        """
        # Count nodes by type
        node_types = {}
        for node in graph_document.nodes:
            node_types[node.type] = node_types.get(node.type, 0) + 1
        
        # Count relationships by type
        rel_types = {}
        for rel in graph_document.relationships:
            rel_types[rel.type] = rel_types.get(rel.type, 0) + 1
        
        # Count connected vs isolated nodes
        connected_nodes = set()
        for rel in graph_document.relationships:
            connected_nodes.add(rel.source.id)
            connected_nodes.add(rel.target.id)
        
        isolated_nodes = len(graph_document.nodes) - len(connected_nodes)
        
        return {
            "total_nodes": len(graph_document.nodes),
            "total_relationships": len(graph_document.relationships),
            "connected_nodes": len(connected_nodes),
            "isolated_nodes": isolated_nodes,
            "node_types": node_types,
            "relationship_types": rel_types,
        }
