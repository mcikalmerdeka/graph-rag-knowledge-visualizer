"""Neo4j graph database client for storing and querying knowledge graphs.

This module provides a wrapper around LangChain's Neo4jGraph for
storing and retrieving knowledge graphs from a Neo4j database.
"""

from typing import List, Optional, Dict, Any

from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document

from src.config.settings import settings
from src.config.logging_config import logger_transformer
from src.models.graph_models import GraphDocument, Node, Relationship
from src.exceptions.custom_exceptions import ConfigurationError, DatabaseError


class Neo4jGraphClient:
    """A client for storing and querying knowledge graphs in Neo4j.
    
    This class wraps LangChain's Neo4jGraph and provides a simplified
    interface for persisting knowledge graphs to a Neo4j database.
    
    Features:
    - Graph document storage
    - Cypher query execution
    - Schema management
    - Connection management
    
    Example:
        >>> client = Neo4jGraphClient()
        >>> # Store graph documents
        >>> client.add_graph_documents(graph_documents)
        >>> # Query the graph
        >>> results = client.query("MATCH (n) RETURN n LIMIT 10")
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize the Neo4jGraphClient.
        
        Args:
            url: Neo4j database URI (default from settings)
            username: Neo4j username (default from settings)
            password: Neo4j password (default from settings)
            database: Database name (default from settings)
            timeout: Transaction timeout in seconds (default from settings)
        
        Raises:
            ConfigurationError: If Neo4j password is not configured
            DatabaseError: If connection to Neo4j fails
        """
        logger_transformer.info("Initializing Neo4jGraphClient")
        
        # Validate configuration
        try:
            settings.validate_neo4j()
            logger_transformer.debug("Neo4j settings validation passed")
        except ValueError as e:
            logger_transformer.error(f"Neo4j settings validation failed: {str(e)}")
            raise ConfigurationError(str(e))
        
        # Set connection parameters
        self.url = url or settings.NEO4J_URI
        self.username = username or settings.NEO4J_USERNAME
        self.password = password or settings.NEO4J_PASSWORD
        self.database = database or settings.NEO4J_DATABASE
        self.timeout = timeout or settings.NEO4J_TIMEOUT
        
        # Initialize graph connection (will be created on first use)
        self._graph: Optional[Neo4jGraph] = None
        
        logger_transformer.debug(f"Neo4jGraphClient configured: url={self.url}, database={self.database}")
    
    def _get_graph(self) -> Neo4jGraph:
        """Get or create the Neo4jGraph instance.
        
        Returns:
            Neo4jGraph instance
        
        Raises:
            DatabaseError: If connection fails
        """
        if self._graph is None:
            try:
                logger_transformer.debug("Creating Neo4jGraph connection")
                self._graph = Neo4jGraph(
                    url=self.url,
                    username=self.username,
                    password=self.password,
                    database=self.database,
                    timeout=self.timeout,
                )
                logger_transformer.info("Neo4jGraph connection established successfully")
            except Exception as e:
                logger_transformer.error(f"Failed to connect to Neo4j: {str(e)}", exc_info=True)
                raise DatabaseError(f"Failed to connect to Neo4j: {str(e)}")
        
        return self._graph
    
    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
    ) -> None:
        """Store graph documents in Neo4j.
        
        This method converts the internal GraphDocument objects to LangChain
        GraphDocument format and stores them in the Neo4j database.
        
        Args:
            graph_documents: List of GraphDocument objects to store
            include_source: Whether to include source document text in the graph
        
        Raises:
            DatabaseError: If storage fails
        """
        if not graph_documents:
            logger_transformer.warning("No graph documents to store")
            return
        
        num_docs = len(graph_documents)
        total_nodes = sum(len(doc.nodes) for doc in graph_documents)
        total_rels = sum(len(doc.relationships) for doc in graph_documents)
        
        logger_transformer.info(
            f"Storing {num_docs} graph document(s) with {total_nodes} nodes and "
            f"{total_rels} relationships in Neo4j"
        )
        
        try:
            graph = self._get_graph()
            
            # Convert internal GraphDocument to LangChain GraphDocument
            lc_documents = self._convert_to_lc_graph_documents(graph_documents)
            
            # Add to Neo4j
            graph.add_graph_documents(
                lc_documents,
                include_source=include_source,
            )
            
            logger_transformer.info("Graph documents successfully stored in Neo4j")
            
        except Exception as e:
            logger_transformer.error(f"Failed to store graph documents: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to store graph documents: {str(e)}")
    
    def query(
        self,
        cypher_query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query against the Neo4j database.
        
        Args:
            cypher_query: The Cypher query to execute
            params: Optional parameters for the query
        
        Returns:
            List of dictionaries containing query results
        
        Raises:
            DatabaseError: If query execution fails
        """
        logger_transformer.debug(f"Executing Cypher query: {cypher_query[:100]}...")
        
        try:
            graph = self._get_graph()
            results = graph.query(cypher_query, params or {})
            logger_transformer.debug(f"Query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger_transformer.error(f"Query execution failed: {str(e)}", exc_info=True)
            raise DatabaseError(f"Query execution failed: {str(e)}")
    
    def refresh_schema(self) -> None:
        """Refresh the Neo4j graph schema information.
        
        This updates the cached schema information used for query generation.
        """
        try:
            graph = self._get_graph()
            graph.refresh_schema()
            logger_transformer.info("Neo4j schema refreshed successfully")
        except Exception as e:
            logger_transformer.error(f"Failed to refresh schema: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to refresh schema: {str(e)}")
    
    def get_schema(self) -> str:
        """Get the current schema of the Neo4j graph.
        
        Returns:
            String representation of the graph schema
        """
        try:
            graph = self._get_graph()
            return graph.get_schema
        except Exception as e:
            logger_transformer.error(f"Failed to get schema: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get schema: {str(e)}")
    
    def clear_graph(self, confirm: bool = False) -> None:
        """Clear all data from the Neo4j graph.
        
        WARNING: This is a destructive operation!
        
        Args:
            confirm: Must be True to actually clear the graph
        
        Raises:
            ValueError: If confirm is not True
            DatabaseError: If clearing fails
        """
        if not confirm:
            raise ValueError("Must set confirm=True to clear the graph. This is a destructive operation!")
        
        logger_transformer.warning("Clearing all data from Neo4j graph")
        
        try:
            graph = self._get_graph()
            # Delete all nodes and relationships
            graph.query("MATCH (n) DETACH DELETE n")
            logger_transformer.info("Neo4j graph cleared successfully")
        except Exception as e:
            logger_transformer.error(f"Failed to clear graph: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to clear graph: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Neo4j graph.
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            graph = self._get_graph()
            
            # Count nodes
            node_result = graph.query("MATCH (n) RETURN count(n) as node_count")
            node_count = node_result[0]["node_count"] if node_result else 0
            
            # Count relationships
            rel_result = graph.query("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = rel_result[0]["rel_count"] if rel_result else 0
            
            # Get node labels
            labels_result = graph.query("CALL db.labels() YIELD label RETURN label")
            node_labels = [row["label"] for row in labels_result]
            
            # Get relationship types
            rel_types_result = graph.query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
            rel_types = [row["relationshipType"] for row in rel_types_result]
            
            return {
                "node_count": node_count,
                "relationship_count": rel_count,
                "node_labels": node_labels,
                "relationship_types": rel_types,
            }
            
        except Exception as e:
            logger_transformer.error(f"Failed to get graph stats: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get graph stats: {str(e)}")
    
    def _convert_to_lc_graph_documents(
        self,
        graph_documents: List[GraphDocument],
    ) -> List:
        """Convert internal GraphDocument objects to LangChain GraphDocument format.
        
        Args:
            graph_documents: List of internal GraphDocument objects
        
        Returns:
            List of LangChain GraphDocument objects
        """
        from langchain_community.graphs.graph_document import (
            GraphDocument as LCGraphDocument,
            Node as LCNode,
            Relationship as LCRelationship,
        )
        
        lc_documents = []
        
        for doc in graph_documents:
            # Convert nodes
            lc_nodes = []
            for node in doc.nodes:
                lc_node = LCNode(
                    id=node.id,
                    type=node.type,
                    properties=node.properties,
                )
                lc_nodes.append(lc_node)
            
            # Convert relationships
            lc_relationships = []
            for rel in doc.relationships:
                lc_source = LCNode(
                    id=rel.source.id,
                    type=rel.source.type,
                    properties=rel.source.properties,
                )
                lc_target = LCNode(
                    id=rel.target.id,
                    type=rel.target.type,
                    properties=rel.target.properties,
                )
                
                lc_rel = LCRelationship(
                    source=lc_source,
                    target=lc_target,
                    type=rel.type,
                    properties=rel.properties,
                )
                lc_relationships.append(lc_rel)
            
            # Create source document
            source = Document(page_content=doc.source) if doc.source else None
            
            # Create LangChain GraphDocument
            lc_doc = LCGraphDocument(
                nodes=lc_nodes,
                relationships=lc_relationships,
                source=source,
            )
            lc_documents.append(lc_doc)
        
        return lc_documents
    
    def close(self) -> None:
        """Close the Neo4j connection.
        
        This should be called when you're done using the client.
        """
        if self._graph is not None:
            try:
                self._graph._driver.close()
                logger_transformer.info("Neo4j connection closed")
            except Exception as e:
                logger_transformer.warning(f"Error closing Neo4j connection: {str(e)}")
            finally:
                self._graph = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
