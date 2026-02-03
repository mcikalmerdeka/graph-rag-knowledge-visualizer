"""Data models for the Knowledge Graph Generator."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Node:
    """Represents a node in the knowledge graph."""
    id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"Node(id='{self.id}', type='{self.type}')"


@dataclass
class Relationship:
    """Represents a relationship between two nodes."""
    source: Node
    target: Node
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"Relationship({self.source.id} -> {self.target.id}, type='{self.type}')"


@dataclass
class GraphDocument:
    """Represents a document containing nodes and relationships."""
    nodes: List[Node]
    relationships: List[Relationship]
    source: Optional[str] = None
    
    def __repr__(self) -> str:
        return f"GraphDocument(nodes={len(self.nodes)}, relationships={len(self.relationships)})"
    
    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes if node.type == node_type]
    
    def get_relationships_by_type(self, rel_type: str) -> List[Relationship]:
        """Get all relationships of a specific type."""
        return [rel for rel in self.relationships if rel.type == rel_type]
    
    def validate(self) -> bool:
        """Validate that all relationships reference existing nodes."""
        node_ids = {node.id for node in self.nodes}
        
        for rel in self.relationships:
            if rel.source.id not in node_ids:
                return False
            if rel.target.id not in node_ids:
                return False
        
        return True


@dataclass
class GraphSchema:
    """Schema definition for graph extraction."""
    allowed_nodes: Optional[List[str]] = None
    allowed_relationships: Optional[List[tuple]] = None
    node_properties: Optional[List[str]] = None
    relationship_properties: Optional[List[str]] = None
    strict_mode: bool = True
    
    def to_llm_transformer_kwargs(self) -> Dict[str, Any]:
        """Convert schema to kwargs for LLMGraphTransformer."""
        kwargs = {}
        
        if self.allowed_nodes:
            kwargs["allowed_nodes"] = self.allowed_nodes
        
        if self.allowed_relationships:
            kwargs["allowed_relationships"] = self.allowed_relationships
        
        if self.node_properties:
            kwargs["node_properties"] = self.node_properties
        
        if self.relationship_properties:
            kwargs["relationship_properties"] = self.relationship_properties
        
        kwargs["strict_mode"] = self.strict_mode
        
        return kwargs
