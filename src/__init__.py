from .components import ConversationVectorStore
import os

def initialize_database():
    """Initialize the vector database if it doesn't exist."""
    DB_PATH = "./chroma_db"
    
    if os.path.exists(DB_PATH):
        print(f"Database already exists at {DB_PATH}")
        return False
        
    try:
        # Create vector store
        vector_store = ConversationVectorStore(persist_directory=DB_PATH)
        
        # Add a test conversation to verify it's working
        test_id = vector_store.add_conversation(
            user_message="This is a test message",
            ai_response="This is a test response",
            metadata={"type": "initialization_test"}
        )
        
        # Verify the test conversation was added
        test_conversation = vector_store.get_conversation_by_id(test_id)
        if test_conversation:
            print("Database initialized successfully!")
            print(f"Database location: {os.path.abspath(DB_PATH)}")
            return True
        else:
            print("Failed to verify database initialization")
            return False
            
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    initialize_database()