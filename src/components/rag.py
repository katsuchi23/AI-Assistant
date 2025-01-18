import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import uuid
from typing import List, Dict, Optional
import math

class ConversationVectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the vector store with ChromaDB."""
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize default embedding function (sentence-transformers)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name="ai_conversations",
            embedding_function=self.embedding_function,
            metadata={"description": "Store for AI model conversations"}
        )

    def add_conversation(
        self,
        user_message: str,
        ai_response: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a conversation pair to the vector store.
        
        Args:
            user_message: The user's input message
            ai_response: The AI's response
            metadata: Additional metadata about the conversation
            
        Returns:
            conversation_id: Unique identifier for the conversation
        """
        # Create a unique ID for the conversation
        conversation_id = str(uuid.uuid4())
        
        # Combine messages for context
        full_context = f"User: {user_message}\nAI: {ai_response}"
        
        # Prepare metadata
        base_metadata = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id
        }
        if metadata:
            base_metadata.update(metadata)
            
        # Add to collection
        self.collection.add(
            documents=[full_context],
            metadatas=[base_metadata],
            ids=[conversation_id]
        )
        
        return conversation_id

    def search_conversations(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Search for similar conversations.
        
        Args:
            query: The search query
            n_results: Number of results to return
            
        Returns:
            List of similar conversations with their metadata
        """
        max_results = 30
        scale_factor = 0.5
        count = self.collection.count()
        n_result = 3 + int(scale_factor * math.log2(max(1, count)))

        if n_result > max_results:
            n_result = max_results

        results = self.collection.query(
            query_texts=[query],
            n_results=n_result
        )
        
        # Format results
        formatted_results = []
        for idx, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            formatted_results.append({
                'content': doc,
                'metadata': metadata,
                'similarity': results['distances'][0][idx] if 'distances' in results else None
            })
            
        return formatted_results

    def get_conversation_by_id(self, conversation_id: str) -> Optional[Dict]:
        """
        Retrieve a specific conversation by ID.
        
        Args:
            conversation_id: The unique identifier of the conversation
            
        Returns:
            Conversation data if found, None otherwise
        """
        try:
            result = self.collection.get(
                ids=[conversation_id]
            )
            if result['documents']:
                return {
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
        except Exception as e:
            print(f"Error retrieving conversation: {e}")
        return None

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation from the store.
        
        Args:
            conversation_id: The unique identifier of the conversation to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[conversation_id])
            return True
        except Exception as e:
            print(f"Error deleting conversation: {e}")
            return False