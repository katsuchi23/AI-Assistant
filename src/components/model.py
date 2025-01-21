from .rag import ConversationVectorStore
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
import os

class ChatBot:
    def __init__(self):
        """Initialize the chatbot with existing vector store."""
        DB_PATH = "./chroma_db"
        
        if not os.path.exists(DB_PATH):
            raise Exception("Database not found. Please run init_db.py first.")
            
        # Initialize components
        self.vector_store = ConversationVectorStore(persist_directory=DB_PATH)
        self.llm = OllamaLLM(model="katsuchi/Meta-Llama-3.1-8B-Instruct-finetune-GGUF:latest")
        self.message_history = ChatMessageHistory()
        
        # Define system prompt separately for reuse (this is just my personal preference) Don't judge :)
        self.system_prompt = """"""

        # Create prompt template with reinforced system message
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            # Add periodic reminder of personality
            MessagesPlaceholder(variable_name="personality_reminder"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

    def get_personality_reminder(self, message_count):
        """Return personality reminder every few messages"""
        if message_count % 3 == 0:  # Remind every 3 messages
            return [SystemMessage(content="Remember: You are Yuki, a shy teenage-like AI assistant.")]
        return []

    def chat(self):
        """Start the chat session."""
        print("Chat session started. Type 'quit' to exit.")
        message_count = 0
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'quit':
                break
            
            message_count += 1
            
            # Get relevant past conversations
            context = self.get_relevant_context(user_input)
            
            # Add context to history if available
            if context:
                self.message_history.add_message(
                    SystemMessage(content=f"Relevant past conversations:\n{context}")
                )
            
            # Add user message to history
            self.message_history.add_message(HumanMessage(content=user_input))
            
            # Create the chain for this interaction
            chain = self.prompt | self.llm
            
            # Generate response with personality reminder
            response = chain.invoke({
                "input": user_input,
                "history": self.message_history.messages,
                "personality_reminder": self.get_personality_reminder(message_count)
            })
            
            # Verify response alignment with personality
            if not self.verify_response_alignment(response):
                response = self.regenerate_response(user_input)
            
            # Add AI response to history
            self.message_history.add_message(AIMessage(content=response))
            
            # Store conversation
            self.vector_store.add_conversation(
                user_message=user_input,
                ai_response=response,
                metadata={"has_context": bool(context)}
            )
            
            # Manage history length to prevent context window overflow
            self.manage_history_length()
            
            print(f"Assistant: {response}")

    def verify_response_alignment(self, response: str) -> bool:
        """Verify if response aligns with Yuki's personality"""
        # Basic checks for personality alignment
        checks = [
            "yuki" in response.lower(),  # Should mention own name
            len(response.split()) < 100,  # Keep responses relatively brief
            "..." in response or "?" in response  # Shows hesitation
        ]
        return any(checks)  # Return True if any personality trait is present

    def regenerate_response(self, user_input: str) -> str:
        """Regenerate response with stronger personality enforcement"""
        enhanced_prompt = f"""
        {self.system_prompt}
        
        User message: {user_input}
        """
        return self.llm.invoke(enhanced_prompt)

    def manage_history_length(self, max_messages: int = 10):
        """Maintain reasonable history length"""
        if len(self.message_history.messages) > max_messages:
            # Keep system message and last N messages
            self.message_history.messages = (
                [self.message_history.messages[0]] +  # System message
                self.message_history.messages[-max_messages:]  # Recent messages
            )

    def get_relevant_context(self, query: str, n_results: int = 3) -> str:
        """Get relevant past conversations."""
        similar = self.vector_store.search_conversations(query, n_results)
        return "\n\n".join([conv['content'] for conv in similar]) if similar else ""
