"""Main entry point for the Knowledge Graph Generator.

This module provides a high-level interface for extracting knowledge graphs
from text and visualizing them.

Usage:
    from src.main import KnowledgeGraphGenerator
    
    generator = KnowledgeGraphGenerator()
    graph = generator.generate("Your text here")
    generator.visualize(graph)
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Union

from src.config.settings import settings
from src.core.graph_transformer import GraphTransformer
from src.core.visualization import GraphVisualizer
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
        self.transformer = GraphTransformer(
            model=model,
            temperature=temperature,
            schema=schema,
            ignore_tool_usage=ignore_tool_usage,
        )
        self.visualizer = GraphVisualizer()
    
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
        if isinstance(text, str):
            # Chunk long text if needed
            chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
            chunk_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP
            
            if len(text) > chunk_size:
                texts = chunk_text(text, chunk_size, chunk_overlap)
            else:
                texts = [text]
        else:
            texts = text
        
        return await self.transformer.extract_graph(texts)
    
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
        return self.visualizer.visualize(graph_document, output_file, **kwargs)
    
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
    generator = KnowledgeGraphGenerator(**kwargs)
    graphs = generator.generate_sync(text)
    
    if visualize and graphs:
        for i, graph in enumerate(graphs):
            if output_file and i == 0:
                file_path = output_file
            else:
                file_path = None
            generator.visualize(graph, file_path)
    
    return graphs


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Marie Curie, born Maria Salomea Skłodowska on 7 November 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields. Her husband, Pierre Curie, was a co-winner of her first Nobel Prize.
    """
    
    print("Generating knowledge graph...")
    graphs = generate_knowledge_graph(
        sample_text,
        output_file="output/sample_graph.html"
    )
    
    print(f"\nGenerated {len(graphs)} graph document(s)")
    
    if graphs:
        generator = KnowledgeGraphGenerator()
        print(generator.display(graphs[0]))
        
        stats = generator.get_stats(graphs)
        print(f"\nStats: {stats}")
