"""Main entry point for the Knowledge Graph Generator.

This module provides a high-level interface for extracting knowledge graphs
from text and visualizing them.

Usage:
    from src.main import KnowledgeGraphGenerator
    
    generator = KnowledgeGraphGenerator()
    graph = generator.generate("Your text here")
    generator.visualize(graph)
    
    # Store in Neo4j
    generator.store_in_neo4j(graph)
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

# Initialize logging at module import
from src.config.logging_config import setup_logger, logger_app, logger_transformer, logger_rag, logger_visualizer
setup_logger()

from src.config.settings import settings
from src.core.graph_transformer import GraphTransformer
from src.core.visualization import GraphVisualizer
from src.core.neo4j_graph import Neo4jGraphClient
from src.models.graph_models import GraphDocument, GraphSchema
from src.utils.helpers import (
    chunk_text,
    format_graph_for_display,
    merge_graph_documents,
    save_graph_to_json,
)


class KnowledgeGraphGenerator:
    """
    High-level interface for knowledge graph generation.
    
    This class provides a simple interface for:
    1. Extracting knowledge graphs from text
    2. Visualizing the graphs
    3. Saving graphs to various formats
    
    Example:
        >>> generator = KnowledgeGraphGenerator()
        >>> 
        >>> # Simple usage
        >>> graph = generator.generate("Albert Einstein developed relativity.")
        >>> generator.visualize(graph)
        >>> 
        >>> # With schema for consistent extraction
        >>> schema = GraphSchema(
        ...     allowed_nodes=["Person", "Organization", "Achievement"],
        ...     allowed_relationships=[("Person", "ACHIEVED", "Achievement")],
        ... )
        >>> generator = KnowledgeGraphGenerator(schema=schema)
        >>> graph = generator.generate("Marie Curie won the Nobel Prize.")
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        schema: Optional[GraphSchema] = None,
        ignore_tool_usage: bool = False,
    ):
        """
        Initialize the KnowledgeGraphGenerator.
        
        Args:
            model: LLM model name
            temperature: Temperature for LLM
            schema: Graph schema for consistent extractions
            ignore_tool_usage: Force prompt-based extraction
        """
        logger_app.info(f"Initializing KnowledgeGraphGenerator with model={model or settings.DEFAULT_LLM_MODEL}")
        
        self.transformer = GraphTransformer(
            model=model,
            temperature=temperature,
            schema=schema,
            ignore_tool_usage=ignore_tool_usage,
        )
        self.visualizer = GraphVisualizer()
        
        logger_app.debug("KnowledgeGraphGenerator initialized successfully")
    
    async def generate(
        self,
        text: Union[str, List[str]],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[GraphDocument]:
        """
        Generate knowledge graph(s) from text.
        
        Args:
            text: Single text or list of texts to process
            chunk_size: Size for text chunking (optional)
            chunk_overlap: Overlap for text chunking (optional)
        
        Returns:
            List of GraphDocument objects
        """
        logger_transformer.info("Starting graph generation from text")
        
        if isinstance(text, str):
            text_length = len(text)
            logger_transformer.debug(f"Processing single text input (length: {text_length})")
            
            # Chunk long text if needed
            chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
            chunk_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP
            
            if text_length > chunk_size:
                texts = chunk_text(text, chunk_size, chunk_overlap)
                logger_transformer.info(f"Text chunked into {len(texts)} chunks (size={chunk_size}, overlap={chunk_overlap})")
            else:
                texts = [text]
                logger_transformer.debug("Text fits in single chunk, no chunking needed")
        else:
            texts = text
            logger_transformer.info(f"Processing {len(texts)} text inputs")
        
        try:
            result = await self.transformer.extract_graph(texts)
            total_nodes = sum(len(doc.nodes) for doc in result)
            total_rels = sum(len(doc.relationships) for doc in result)
            logger_transformer.info(f"Graph extraction completed: {len(result)} document(s), {total_nodes} total nodes, {total_rels} total relationships")
            return result
        except Exception as e:
            logger_transformer.error(f"Graph extraction failed: {str(e)}", exc_info=True)
            raise
    
    def generate_sync(
        self,
        text: Union[str, List[str]],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[GraphDocument]:
        """
        Synchronous version of generate.
        
        Args:
            text: Single text or list of texts to process
            chunk_size: Size for text chunking (optional)
            chunk_overlap: Overlap for text chunking (optional)
        
        Returns:
            List of GraphDocument objects
        """
        return asyncio.run(self.generate(text, chunk_size, chunk_overlap))
    
    def visualize(
        self,
        graph_document: GraphDocument,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Visualize a knowledge graph.
        
        Args:
            graph_document: GraphDocument to visualize
            output_file: Output file path (default: auto-generated)
            **kwargs: Additional arguments for visualizer
        """
        num_nodes = len(graph_document.nodes)
        num_rels = len(graph_document.relationships)
        logger_visualizer.info(f"Visualizing graph: {num_nodes} nodes, {num_rels} relationships")
        
        try:
            result = self.visualizer.visualize(graph_document, output_file, **kwargs)
            logger_visualizer.info(f"Graph visualization saved to {output_file or 'default location'}")
            return result
        except Exception as e:
            logger_visualizer.error(f"Graph visualization failed: {str(e)}", exc_info=True)
            raise
    
    def visualize_all(
        self,
        graph_documents: List[GraphDocument],
        output_dir: Optional[Union[str, Path]] = None,
        prefix: str = "graph",
    ):
        """
        Visualize multiple graph documents.
        
        Args:
            graph_documents: List of GraphDocuments
            output_dir: Output directory
            prefix: Filename prefix
        """
        return self.visualizer.visualize_batch(
            graph_documents,
            output_dir=output_dir,
            prefix=prefix,
        )
    
    def save(
        self,
        graph_document: GraphDocument,
        output_file: Union[str, Path],
    ) -> Path:
        """
        Save a graph to JSON format.
        
        Args:
            graph_document: GraphDocument to save
            output_file: Output file path
        
        Returns:
            Path to saved file
        """
        return save_graph_to_json(graph_document, output_file)
    
    def display(self, graph_document: GraphDocument) -> str:
        """
        Get a formatted string representation of the graph.
        
        Args:
            graph_document: GraphDocument to format
        
        Returns:
            Formatted string
        """
        return format_graph_for_display(graph_document)
    
    def merge(self, graph_documents: List[GraphDocument]) -> GraphDocument:
        """
        Merge multiple graph documents into one.
        
        Args:
            graph_documents: List of GraphDocuments
        
        Returns:
            Merged GraphDocument
        """
        return merge_graph_documents(graph_documents)
    
    def get_stats(self, graph_documents: List[GraphDocument]) -> dict:
        """
        Get statistics about extracted graphs.
        
        Args:
            graph_documents: List of GraphDocuments
        
        Returns:
            Dictionary with statistics
        """
        return self.transformer.get_stats(graph_documents)
    
    def store_in_neo4j(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
    ) -> None:
        """Store graph documents in Neo4j database.
        
        Args:
            graph_documents: List of GraphDocuments to store
            include_source: Whether to include source document text in the graph
        
        Example:
            >>> generator = KnowledgeGraphGenerator()
            >>> graphs = generator.generate_sync("Your text here")
            >>> generator.store_in_neo4j(graphs, include_source=True)
        """
        logger_app.info(f"Storing {len(graph_documents)} graph(s) in Neo4j")
        
        try:
            self.transformer.store_in_neo4j_sync(
                graph_documents,
                include_source=include_source,
            )
            logger_app.info("Graphs successfully stored in Neo4j")
        except Exception as e:
            logger_app.error(f"Failed to store graphs in Neo4j: {str(e)}", exc_info=True)
            raise
    
    def query_neo4j(self, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query against the Neo4j database.
        
        Args:
            cypher_query: The Cypher query to execute
            params: Optional parameters for the query
        
        Returns:
            List of dictionaries containing query results
        
        Example:
            >>> results = generator.query_neo4j("MATCH (n:Person) RETURN n.name LIMIT 5")
            >>> print(results)
        """
        logger_app.debug(f"Executing Neo4j query: {cypher_query[:100]}...")
        
        try:
            results = self.transformer.query_neo4j(cypher_query, params)
            logger_app.info(f"Query returned {len(results)} results")
            return results
        except Exception as e:
            logger_app.error(f"Neo4j query failed: {str(e)}", exc_info=True)
            raise
    
    def get_neo4j_stats(self) -> Dict[str, Any]:
        """Get statistics about the Neo4j graph.
        
        Returns:
            Dictionary with graph statistics including node count, 
            relationship count, node labels, and relationship types.
        """
        try:
            stats = self.transformer.get_neo4j_stats()
            logger_app.info(f"Neo4j stats: {stats['node_count']} nodes, {stats['relationship_count']} relationships")
            return stats
        except Exception as e:
            logger_app.error(f"Failed to get Neo4j stats: {str(e)}", exc_info=True)
            raise
    
    def close(self) -> None:
        """Close resources and connections.
        
        This should be called when you're done using the generator to ensure
        all connections are properly closed.
        """
        logger_app.info("Closing KnowledgeGraphGenerator resources")
        self.transformer.close_neo4j()


# Convenience function for quick usage
def generate_knowledge_graph(
    text: str,
    visualize: bool = True,
    output_file: Optional[str] = None,
    **kwargs,
) -> List[GraphDocument]:
    """
    Quick function to generate a knowledge graph from text.
    
    Args:
        text: Text to process
        visualize: Whether to create visualization
        output_file: Output file path for visualization
        **kwargs: Additional arguments for KnowledgeGraphGenerator
    
    Returns:
        List of GraphDocument objects
    
    Example:
        >>> from src.main import generate_knowledge_graph
        >>> graphs = generate_knowledge_graph(
        ...     "Marie Curie won the Nobel Prize in Physics.",
        ...     output_file="marie_curie.html"
        ... )
    """
    logger_app.info("Starting knowledge graph generation workflow")
    
    try:
        generator = KnowledgeGraphGenerator(**kwargs)
        graphs = generator.generate_sync(text)
        
        logger_app.info(f"Successfully generated {len(graphs)} graph document(s)")
        
        if visualize and graphs:
            logger_app.info(f"Creating visualizations for {len(graphs)} graph(s)")
            for i, graph in enumerate(graphs):
                if output_file and i == 0:
                    file_path = output_file
                else:
                    file_path = None
                generator.visualize(graph, file_path)
        
        return graphs
    except Exception as e:
        logger_app.error(f"Knowledge graph generation workflow failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Marie Curie, born Maria Salomea Skłodowska on 7 November 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields. Her husband, Pierre Curie, was a co-winner of her first Nobel Prize.
    """
    
    logger_app.info("Running example usage in __main__")
    
    try:
        generator = KnowledgeGraphGenerator()
        
        # Generate graphs from text
        graphs = generator.generate_sync(sample_text)
        logger_app.info(f"Generated {len(graphs)} graph document(s)")
        
        if graphs:
            # Display the first graph
            print(generator.display(graphs[0]))
            
            # Get statistics
            stats = generator.get_stats(graphs)
            logger_app.info(f"Graph stats: {stats}")
            
            # Create visualization
            generator.visualize(graphs[0], output_file="output/sample_graph.html")
            
            # Store in Neo4j (if configured)
            try:
                settings.validate_neo4j()
                generator.store_in_neo4j(graphs, include_source=True)
                
                # Query Neo4j to verify
                neo4j_stats = generator.get_neo4j_stats()
                logger_app.info(f"Neo4j graph stats: {neo4j_stats}")
                
                # Example query
                results = generator.query_neo4j("MATCH (n) RETURN n.id, n.type LIMIT 5")
                logger_app.info(f"Sample nodes from Neo4j: {results}")
            except ValueError:
                logger_app.warning("Neo4j not configured, skipping database storage")
        
        # Close resources
        generator.close()
        
    except Exception as e:
        logger_app.error(f"Example usage failed: {str(e)}", exc_info=True)
        raise
