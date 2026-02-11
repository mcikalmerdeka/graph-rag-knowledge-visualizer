"""Graph transformation and extraction using LLM."""

import asyncio
from typing import List, Optional, Union

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

from src.config.settings import settings
from src.config.logging_config import logger_transformer
from src.models.graph_models import GraphDocument, Node, Relationship, GraphSchema
from src.core.neo4j_graph import Neo4jGraphClient
from src.exceptions.custom_exceptions import (
    ConfigurationError,
    ExtractionError,
    LLMError,
    DatabaseError,
)


class GraphTransformer:
    """
    A class for transforming text into knowledge graphs using LLMs.
    
    This class wraps LangChain's LLMGraphTransformer and provides a simplified
    interface for extracting structured knowledge graphs from text.
    
    Features:
    - Async processing for better performance
    - Configurable graph schema for consistent extractions
    - Property extraction for richer graphs
    - Support for both tool-based and prompt-based extraction
    
    Example:
        >>> transformer = GraphTransformer()
        >>> documents = ["Marie Curie won the Nobel Prize in Physics."]
        >>> graphs = await transformer.extract_graph(documents)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        schema: Optional[GraphSchema] = None,
        ignore_tool_usage: bool = False,
    ):
        """
        Initialize the GraphTransformer.
        
        Args:
            model: LLM model name (default from settings)
            temperature: Temperature for LLM (default from settings)
            schema: Graph schema for consistent extractions (optional)
            ignore_tool_usage: Force prompt-based extraction even if tools are available
        
        Raises:
            ConfigurationError: If OpenAI API key is not configured
        """
        logger_transformer.info(f"Initializing GraphTransformer with model={model or settings.DEFAULT_LLM_MODEL}")
        
        # Validate configuration
        try:
            settings.validate()
            logger_transformer.debug("Settings validation passed")
        except ValueError as e:
            logger_transformer.error(f"Settings validation failed: {str(e)}")
            raise ConfigurationError(str(e))
        
        # Initialize LLM
        self.model_name = model or settings.DEFAULT_LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.DEFAULT_TEMPERATURE
        
        try:
            self.llm = ChatOpenAI(
                temperature=self.temperature,
                model_name=self.model_name,
                api_key=settings.OPENAI_API_KEY,
            )
            logger_transformer.debug(f"LLM initialized: model={self.model_name}, temperature={self.temperature}")
        except Exception as e:
            logger_transformer.error(f"Failed to initialize LLM: {str(e)}", exc_info=True)
            raise LLMError(f"Failed to initialize LLM: {str(e)}")
        
        # Store schema
        self.schema = schema
        self.ignore_tool_usage = ignore_tool_usage
        
        if self.schema:
            logger_transformer.info("Using custom graph schema for extraction")
        
        # Initialize transformer (will be created on first use)
        self._transformer: Optional[LLMGraphTransformer] = None
        
        # Initialize Neo4j client (lazy initialization)
        self._neo4j_client: Optional[Neo4jGraphClient] = None
    
    def _get_transformer(self) -> LLMGraphTransformer:
        """Get or create the LLMGraphTransformer instance."""
        if self._transformer is None:
            kwargs = {
                "llm": self.llm,
                "ignore_tool_usage": self.ignore_tool_usage,
            }
            
            # Add schema configuration if provided
            if self.schema:
                kwargs.update(self.schema.to_llm_transformer_kwargs())
            
            self._transformer = LLMGraphTransformer(**kwargs)
        
        return self._transformer
    
    async def extract_graph(
        self,
        texts: Union[str, List[str]],
    ) -> List[GraphDocument]:
        """
        Extract knowledge graph from text(s).
        
        Args:
            texts: Single text string or list of text strings to process
        
        Returns:
            List of GraphDocument objects containing nodes and relationships
        
        Raises:
            ExtractionError: If extraction fails
        
        Example:
            >>> texts = ["Albert Einstein developed the theory of relativity."]
            >>> graphs = await transformer.extract_graph(texts)
            >>> print(graphs[0].nodes)
        """
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        
        num_docs = len(texts)
        logger_transformer.info(f"Extracting graph from {num_docs} document(s)")
        
        # Create Document objects
        documents = [Document(page_content=text) for text in texts]
        
        try:
            transformer = self._get_transformer()
            
            # Use async conversion for better performance
            logger_transformer.debug("Calling LLMGraphTransformer for extraction")
            graph_documents = await transformer.aconvert_to_graph_documents(documents)
            
            # Convert to our model format
            result = self._convert_to_graph_documents(graph_documents)
            
            total_nodes = sum(len(doc.nodes) for doc in result)
            total_rels = sum(len(doc.relationships) for doc in result)
            logger_transformer.info(f"Graph extraction successful: {len(result)} document(s), {total_nodes} nodes, {total_rels} relationships")
            
            return result
            
        except Exception as e:
            logger_transformer.error(f"Graph extraction failed: {str(e)}", exc_info=True)
            raise ExtractionError(f"Failed to extract graph: {str(e)}")
    
    def extract_graph_sync(
        self,
        texts: Union[str, List[str]],
    ) -> List[GraphDocument]:
        """
        Synchronous version of extract_graph.
        
        Args:
            texts: Single text string or list of text strings to process
        
        Returns:
            List of GraphDocument objects containing nodes and relationships
        """
        return asyncio.run(self.extract_graph(texts))
    
    def _convert_to_graph_documents(
        self,
        raw_documents,
    ) -> List[GraphDocument]:
        """
        Convert raw LangChain GraphDocument objects to our model format.
        
        Args:
            raw_documents: List of LangChain GraphDocument objects
        
        Returns:
            List of our GraphDocument model objects
        """
        graph_documents = []
        
        for doc in raw_documents:
            # Convert nodes
            nodes = []
            for raw_node in doc.nodes:
                node = Node(
                    id=raw_node.id,
                    type=raw_node.type,
                    properties=dict(raw_node.properties) if raw_node.properties else {},
                )
                nodes.append(node)
            
            # Convert relationships
            relationships = []
            for raw_rel in doc.relationships:
                source = Node(
                    id=raw_rel.source.id,
                    type=raw_rel.source.type,
                    properties=dict(raw_rel.source.properties) if raw_rel.source.properties else {},
                )
                target = Node(
                    id=raw_rel.target.id,
                    type=raw_rel.target.type,
                    properties=dict(raw_rel.target.properties) if raw_rel.target.properties else {},
                )
                
                rel = Relationship(
                    source=source,
                    target=target,
                    type=raw_rel.type,
                    properties=dict(raw_rel.properties) if raw_rel.properties else {},
                )
                relationships.append(rel)
            
            # Create GraphDocument
            graph_doc = GraphDocument(
                nodes=nodes,
                relationships=relationships,
                source=doc.source.page_content if doc.source else None,
            )
            
            graph_documents.append(graph_doc)
        
        return graph_documents
    
    def get_stats(self, graph_documents: List[GraphDocument]) -> dict:
        """
        Get statistics about extracted graph documents.
        
        Args:
            graph_documents: List of GraphDocument objects
        
        Returns:
            Dictionary containing statistics
        """
        total_nodes = sum(len(doc.nodes) for doc in graph_documents)
        total_relationships = sum(len(doc.relationships) for doc in graph_documents)
        
        # Count unique node types
        node_types = set()
        for doc in graph_documents:
            for node in doc.nodes:
                node_types.add(node.type)
        
        # Count unique relationship types
        rel_types = set()
        for doc in graph_documents:
            for rel in doc.relationships:
                rel_types.add(rel.type)
        
        return {
            "num_documents": len(graph_documents),
            "total_nodes": total_nodes,
            "total_relationships": total_relationships,
            "avg_nodes_per_doc": total_nodes / len(graph_documents) if graph_documents else 0,
            "avg_rels_per_doc": total_relationships / len(graph_documents) if graph_documents else 0,
            "unique_node_types": len(node_types),
            "unique_relationship_types": len(rel_types),
            "node_types": sorted(list(node_types)),
            "relationship_types": sorted(list(rel_types)),
        }
    
    def _get_neo4j_client(self) -> Neo4jGraphClient:
        """Get or create the Neo4jGraphClient instance."""
        if self._neo4j_client is None:
            logger_transformer.debug("Initializing Neo4jGraphClient")
            self._neo4j_client = Neo4jGraphClient()
        return self._neo4j_client
    
    async def store_in_neo4j(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
    ) -> None:
        """Store graph documents in Neo4j database.
        
        Args:
            graph_documents: List of GraphDocument objects to store
            include_source: Whether to include source document text in the graph
        
        Raises:
            DatabaseError: If storage fails
        """
        logger_transformer.info(f"Storing {len(graph_documents)} graph document(s) in Neo4j")
        
        try:
            client = self._get_neo4j_client()
            client.add_graph_documents(
                graph_documents,
                include_source=include_source,
            )
            logger_transformer.info("Successfully stored graph documents in Neo4j")
        except Exception as e:
            logger_transformer.error(f"Failed to store graph in Neo4j: {str(e)}", exc_info=True)
            raise
    
    def store_in_neo4j_sync(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
    ) -> None:
        """Synchronous version of store_in_neo4j.
        
        Args:
            graph_documents: List of GraphDocument objects to store
            include_source: Whether to include source document text in the graph
        """
        return asyncio.run(self.store_in_neo4j(graph_documents, include_source))
    
    def query_neo4j(self, cypher_query: str, params: Optional[dict] = None) -> List[dict]:
        """Execute a Cypher query against Neo4j.
        
        Args:
            cypher_query: The Cypher query to execute
            params: Optional parameters for the query
        
        Returns:
            List of dictionaries containing query results
        """
        try:
            client = self._get_neo4j_client()
            return client.query(cypher_query, params)
        except Exception as e:
            logger_transformer.error(f"Neo4j query failed: {str(e)}", exc_info=True)
            raise
    
    def get_neo4j_stats(self) -> dict:
        """Get statistics about the Neo4j graph.
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            client = self._get_neo4j_client()
            return client.get_stats()
        except Exception as e:
            logger_transformer.error(f"Failed to get Neo4j stats: {str(e)}", exc_info=True)
            raise
    
    def close_neo4j(self) -> None:
        """Close the Neo4j connection."""
        if self._neo4j_client is not None:
            self._neo4j_client.close()
            self._neo4j_client = None
            logger_transformer.info("Neo4j connection closed")
