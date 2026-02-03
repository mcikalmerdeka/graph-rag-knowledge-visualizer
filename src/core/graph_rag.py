"""Graph RAG implementation for knowledge graph-based Q&A."""

import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import networkx as nx

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.models.graph_models import GraphDocument, Node, Relationship
from src.config.settings import settings


class GraphRAG:
    """
    Graph-based RAG system for querying knowledge graphs with natural language.
    
    This class combines graph traversal with LLM generation to answer questions
    about the extracted knowledge graph.
    
    Features:
    - Semantic search over nodes and relationships
    - Graph traversal for multi-hop reasoning
    - Context extraction for LLM
    - Natural language query processing
    
    Example:
        >>> graph_doc = generator.generate_sync(text)[0]
        >>> rag = GraphRAG(graph_doc)
        >>> answer = rag.query("What are the side effects of metformin?")
    """
    
    def __init__(
        self,
        graph_document: GraphDocument,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """
        Initialize GraphRAG with a knowledge graph.
        
        Args:
            graph_document: GraphDocument containing nodes and relationships
            model: LLM model name
            temperature: Temperature for LLM generation
        """
        self.graph_document = graph_document
        self.model_name = model or settings.DEFAULT_LLM_MODEL
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=self.temperature,
            model_name=self.model_name,
            api_key=settings.OPENAI_API_KEY,
        )
        
        # Build NetworkX graph
        self.graph = self._build_networkx_graph()
        
        # Build keyword index for fast lookup
        self._build_keyword_index()
    
    def _build_networkx_graph(self) -> nx.DiGraph:
        """
        Convert GraphDocument to NetworkX directed graph.
        
        Returns:
            NetworkX DiGraph with node and edge attributes
        """
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.graph_document.nodes:
            G.add_node(
                node.id,
                type=node.type,
                properties=node.properties,
            )
        
        # Add edges
        for rel in self.graph_document.relationships:
            G.add_edge(
                rel.source.id,
                rel.target.id,
                relation=rel.type,
                properties=rel.properties,
            )
        
        return G
    
    def _build_keyword_index(self):
        """Build keyword index for fast node lookup."""
        self.keyword_to_nodes = {}
        
        for node in self.graph_document.nodes:
            # Index node ID
            words = self._extract_keywords(node.id)
            for word in words:
                if word not in self.keyword_to_nodes:
                    self.keyword_to_nodes[word] = []
                self.keyword_to_nodes[word].append(node.id)
            
            # Index properties
            if node.properties:
                for key, value in node.properties.items():
                    if isinstance(value, str):
                        words = self._extract_keywords(value)
                        for word in words:
                            if word not in self.keyword_to_nodes:
                                self.keyword_to_nodes[word] = []
                            if node.id not in self.keyword_to_nodes[word]:
                                self.keyword_to_nodes[word].append(node.id)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for indexing."""
        # Simple keyword extraction - lowercase and split
        text = text.lower()
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text)
        return [w for w in words if len(w) > 2]  # Filter short words
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Find relevant nodes based on query keywords.
        
        Args:
            query: Search query
            top_k: Number of top results to return
        
        Returns:
            List of (node_id, score) tuples
        """
        query_keywords = self._extract_keywords(query)
        node_scores = {}
        
        # Score nodes based on keyword matches
        for keyword in query_keywords:
            if keyword in self.keyword_to_nodes:
                for node_id in self.keyword_to_nodes[keyword]:
                    node_scores[node_id] = node_scores.get(node_id, 0) + 1
        
        # Normalize by query length
        if query_keywords:
            for node_id in node_scores:
                node_scores[node_id] /= len(query_keywords)
        
        # Sort by score
        sorted_nodes = sorted(
            node_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return sorted_nodes[:top_k]
    
    def get_subgraph_context(
        self,
        query: str,
        max_depth: int = 2,
        max_nodes: int = 20,
    ) -> Dict[str, Any]:
        """
        Extract relevant subgraph context for a query.
        
        Args:
            query: User query
            max_depth: Maximum traversal depth from relevant nodes
            max_nodes: Maximum number of nodes to include
        
        Returns:
            Dictionary with relevant nodes, relationships, and paths
        """
        # Find relevant starting nodes
        relevant_nodes = self.semantic_search(query, top_k=5)
        
        if not relevant_nodes:
            return {
                "nodes": [],
                "relationships": [],
                "paths": [],
            }
        
        # Extract subgraph around relevant nodes
        subgraph_nodes = set()
        subgraph_edges = []
        
        for start_node_id, _ in relevant_nodes:
            # BFS to find connected nodes
            visited = {start_node_id}
            queue = [(start_node_id, 0)]
            
            while queue and len(subgraph_nodes) < max_nodes:
                current_id, depth = queue.pop(0)
                
                if depth > max_depth:
                    continue
                
                subgraph_nodes.add(current_id)
                
                # Get neighbors
                for neighbor in self.graph.successors(current_id):
                    edge_data = self.graph.get_edge_data(current_id, neighbor)
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        
                        if len(subgraph_nodes) < max_nodes:
                            subgraph_nodes.add(neighbor)
                    
                    # Add edge
                    subgraph_edges.append({
                        "source": current_id,
                        "target": neighbor,
                        "relation": edge_data.get("relation", "related"),
                        "properties": edge_data.get("properties", {}),
                    })
                
                # Also check predecessors (for undirected-like traversal)
                for predecessor in self.graph.predecessors(current_id):
                    edge_data = self.graph.get_edge_data(predecessor, current_id)
                    
                    if predecessor not in visited:
                        visited.add(predecessor)
                        queue.append((predecessor, depth + 1))
                        
                        if len(subgraph_nodes) < max_nodes:
                            subgraph_nodes.add(predecessor)
                    
                    # Add edge
                    subgraph_edges.append({
                        "source": predecessor,
                        "target": current_id,
                        "relation": edge_data.get("relation", "related"),
                        "properties": edge_data.get("properties", {}),
                    })
        
        # Get node details
        node_details = []
        for node_id in subgraph_nodes:
            node_data = self.graph.nodes[node_id]
            node_details.append({
                "id": node_id,
                "type": node_data.get("type", "unknown"),
                "properties": node_data.get("properties", {}),
            })
        
        # Find paths between relevant nodes
        paths = []
        if len(relevant_nodes) >= 2:
            for i in range(min(3, len(relevant_nodes))):
                for j in range(i + 1, min(3, len(relevant_nodes))):
                    start = relevant_nodes[i][0]
                    end = relevant_nodes[j][0]
                    
                    try:
                        path = nx.shortest_path(self.graph, start, end)
                        paths.append(path)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
        
        return {
            "nodes": node_details,
            "relationships": subgraph_edges,
            "paths": paths,
            "relevant_nodes": [n[0] for n in relevant_nodes],
        }
    
    def format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """
        Format graph context as text for LLM consumption.
        
        Args:
            context: Context dictionary from get_subgraph_context
        
        Returns:
            Formatted context string
        """
        lines = []
        lines.append("Knowledge Graph Context:")
        lines.append("=" * 50)
        
        # Add nodes
        if context["nodes"]:
            lines.append("\nEntities:")
            for node in context["nodes"]:
                props = node.get("properties", {})
                prop_str = ", ".join([f"{k}: {v}" for k, v in props.items()]) if props else ""
                lines.append(f"  - {node['id']} (Type: {node['type']}) {prop_str}")
        
        # Add relationships
        if context["relationships"]:
            lines.append("\nRelationships:")
            seen = set()
            for rel in context["relationships"]:
                key = (rel["source"], rel["relation"], rel["target"])
                if key not in seen:
                    seen.add(key)
                    lines.append(f"  - {rel['source']} --[{rel['relation']}]--> {rel['target']}")
        
        # Add paths
        if context["paths"]:
            lines.append("\nConnection Paths:")
            for i, path in enumerate(context["paths"][:3], 1):
                path_str = " → ".join(path)
                lines.append(f"  Path {i}: {path_str}")
        
        return "\n".join(lines)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the knowledge graph.
        
        Args:
            question: Natural language question
        
        Returns:
            Dictionary with answer, context, and metadata
        """
        # Extract relevant context
        context = self.get_subgraph_context(question)
        
        if not context["nodes"]:
            return {
                "answer": "I don't have enough information in the knowledge graph to answer this question. The graph may not contain relevant entities or relationships.",
                "context": None,
                "confidence": 0.0,
                "sources": [],
            }
        
        # Format context
        context_text = self.format_context_for_llm(context)
        
        # Build prompt
        system_prompt = """You are a knowledgeable assistant that answers questions based on a provided knowledge graph. 
Use the entities and relationships in the context to provide accurate, concise answers.

Guidelines:
- Base your answer ONLY on the provided knowledge graph context
- If the context doesn't contain enough information, say so
- Be specific and cite relevant entities and relationships
- Keep answers clear and well-structured
- If multiple paths or connections exist, mention the most relevant ones"""

        user_prompt = f"""Context:
{context_text}

Question: {question}

Please provide a concise answer based on the knowledge graph context above."""

        # Generate answer
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            # Calculate confidence based on context coverage
            confidence = self._calculate_confidence(question, context)
            
            return {
                "answer": answer,
                "context": context,
                "context_text": context_text,
                "confidence": confidence,
                "sources": context["relevant_nodes"],
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "context": context,
                "context_text": context_text,
                "confidence": 0.0,
                "sources": context["relevant_nodes"],
            }
    
    def _calculate_confidence(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> float:
        """
        Calculate confidence score based on context coverage.
        
        Args:
            query: Original query
            context: Retrieved context
        
        Returns:
            Confidence score between 0 and 1
        """
        query_keywords = set(self._extract_keywords(query))
        
        if not query_keywords:
            return 0.5
        
        # Count how many query keywords appear in context
        context_text = self.format_context_for_llm(context).lower()
        matched_keywords = sum(1 for kw in query_keywords if kw in context_text)
        
        # Base confidence on keyword coverage
        base_confidence = matched_keywords / len(query_keywords)
        
        # Boost if we have paths (multi-hop connections)
        if context.get("paths"):
            base_confidence = min(1.0, base_confidence + 0.1)
        
        # Boost if we have many nodes
        if len(context.get("nodes", [])) > 5:
            base_confidence = min(1.0, base_confidence + 0.05)
        
        return round(base_confidence, 2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": self._count_node_types(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph),
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type."""
        type_counts = {}
        for node_id in self.graph.nodes():
            node_type = self.graph.nodes[node_id].get("type", "unknown")
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts


# Convenience function for quick usage
def create_graph_rag(graph_document: GraphDocument) -> GraphRAG:
    """
    Quick factory function to create a GraphRAG instance.
    
    Args:
        graph_document: GraphDocument to use for RAG
    
    Returns:
        GraphRAG instance
    """
    return GraphRAG(graph_document)
