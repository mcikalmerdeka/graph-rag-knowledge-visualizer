"""Utility functions for the Knowledge Graph Generator."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.models.graph_models import GraphDocument


def save_graph_to_json(
    graph_document: GraphDocument,
    output_file: Union[str, Path],
) -> Path:
    """
    Save a graph document to JSON format.
    
    Args:
        graph_document: GraphDocument to save
        output_file: Path to save the JSON file
    
    Returns:
        Path to the saved file
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    data = {
        "nodes": [
            {
                "id": node.id,
                "type": node.type,
                "properties": node.properties,
            }
            for node in graph_document.nodes
        ],
        "relationships": [
            {
                "source": rel.source.id,
                "target": rel.target.id,
                "type": rel.type,
                "properties": rel.properties,
            }
            for rel in graph_document.relationships
        ],
        "source": graph_document.source,
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return output_file


def load_graph_from_json(input_file: Union[str, Path]) -> GraphDocument:
    """
    Load a graph document from JSON format.
    
    Args:
        input_file: Path to the JSON file
    
    Returns:
        GraphDocument instance
    """
    from src.models.graph_models import Node, Relationship
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Reconstruct nodes
    nodes = [
        Node(id=n["id"], type=n["type"], properties=n.get("properties", {}))
        for n in data["nodes"]
    ]
    
    # Create node lookup for relationships
    node_lookup = {node.id: node for node in nodes}
    
    # Reconstruct relationships
    relationships = []
    for r in data["relationships"]:
        source = node_lookup.get(r["source"])
        target = node_lookup.get(r["target"])
        if source and target:
            rel = Relationship(
                source=source,
                target=target,
                type=r["type"],
                properties=r.get("properties", {}),
            )
            relationships.append(rel)
    
    return GraphDocument(
        nodes=nodes,
        relationships=relationships,
        source=data.get("source"),
    )


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Split text into chunks for processing.
    
    Simple implementation for text chunking. For production use,
    consider using RecursiveCharacterTextSplitter from langchain.
    
    Args:
        text: Text to split
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence ending
            for char in ['. ', '! ', '? ', '\n\n']:
                pos = text.rfind(char, start, end)
                if pos != -1:
                    end = pos + len(char)
                    break
            else:
                # Look for word boundary
                pos = text.rfind(' ', start, end)
                if pos != -1:
                    end = pos + 1
        
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap
    
    return chunks


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string to be used as a filename.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    return filename or "unnamed"


def format_graph_for_display(graph_document: GraphDocument) -> str:
    """
    Format a graph document for human-readable display.
    
    Args:
        graph_document: GraphDocument to format
    
    Returns:
        Formatted string representation
    """
    lines = []
    lines.append("=" * 60)
    lines.append("KNOWLEDGE GRAPH")
    lines.append("=" * 60)
    
    # Nodes section
    lines.append(f"\nNODES ({len(graph_document.nodes)} total):")
    lines.append("-" * 60)
    
    # Group nodes by type
    nodes_by_type = {}
    for node in graph_document.nodes:
        if node.type not in nodes_by_type:
            nodes_by_type[node.type] = []
        nodes_by_type[node.type].append(node)
    
    for node_type, nodes in sorted(nodes_by_type.items()):
        lines.append(f"\n  {node_type} ({len(nodes)}):")
        for node in nodes:
            lines.append(f"    • {node.id}")
            if node.properties:
                for key, value in node.properties.items():
                    lines.append(f"      - {key}: {value}")
    
    # Relationships section
    lines.append(f"\n\nRELATIONSHIPS ({len(graph_document.relationships)} total):")
    lines.append("-" * 60)
    
    # Group relationships by type
    rels_by_type = {}
    for rel in graph_document.relationships:
        if rel.type not in rels_by_type:
            rels_by_type[rel.type] = []
        rels_by_type[rel.type].append(rel)
    
    for rel_type, rels in sorted(rels_by_type.items()):
        lines.append(f"\n  {rel_type} ({len(rels)}):")
        for rel in rels:
            lines.append(f"    • {rel.source.id} → {rel.target.id}")
            if rel.properties:
                for key, value in rel.properties.items():
                    lines.append(f"      - {key}: {value}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


def merge_graph_documents(documents: List[GraphDocument]) -> GraphDocument:
    """
    Merge multiple graph documents into a single document.
    
    This is useful when processing multiple texts and combining results.
    
    Args:
        documents: List of GraphDocument objects
    
    Returns:
        Merged GraphDocument
    """
    all_nodes = {}
    all_relationships = []
    
    for doc in documents:
        # Add nodes (use dict to avoid duplicates)
        for node in doc.nodes:
            if node.id not in all_nodes:
                all_nodes[node.id] = node
        
        # Add relationships
        for rel in doc.relationships:
            all_relationships.append(rel)
    
    return GraphDocument(
        nodes=list(all_nodes.values()),
        relationships=all_relationships,
        source=None,  # Merged from multiple sources
    )
