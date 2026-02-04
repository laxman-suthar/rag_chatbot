"""
RAG (Retrieval-Augmented Generation) Engine
Core logic for document retrieval and response generation
 FIXED VERSION - Gemini-only
"""

import logging
import time
from typing import List, Dict, Tuple, Optional
from django.conf import settings
from .vector_store import VectorStore
from .embeddings import EmbeddingModel

logger = logging.getLogger('rag_service')


class RAGEngine:
    """
    Main RAG Engine for handling queries and generating responses
    """
    
    def __init__(self):
        """Initialize RAG engine with vector store and LLM client"""
        self.vector_store = VectorStore()
        self.embedding_model = EmbeddingModel()
        
        # Initialize LLM client based on provider
        if settings.LLM_CONFIG['provider'] == 'gemini':
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.llm_client = genai
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {str(e)}")
                raise
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.LLM_CONFIG['provider']}")
        
        self.config = settings.RAG_CONFIG
        self.llm_config = settings.LLM_CONFIG
        
        logger.info("RAG Engine initialized successfully")
    
    def get_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
        top_k: Optional[int] = None
    ) -> Dict:
        """
        Get response for a user query using RAG
        
        Args:
            query: User's question
            conversation_history: Previous conversation messages
            top_k: Number of chunks to retrieve (overrides config)
        
        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant context
            retrieval_start = time.time()
            retrieved_chunks = self.retrieve_context(query, top_k)
            retrieval_time = time.time() - retrieval_start
            
            if not retrieved_chunks:
                logger.warning(f"No relevant context found for query: {query[:50]}...")
                return {
                    'response': "I apologize, but I couldn't find relevant information in the knowledge base to answer your question. Could you please rephrase or provide more details?",
                    'sources': [],
                    'retrieved_chunks': [],
                    'retrieval_time': retrieval_time,
                    'llm_time': 0,
                    'total_time': time.time() - start_time
                }
            
            # Step 2: Generate response using LLM
            llm_start = time.time()
            response, sources = self.generate_response(
                query,
                retrieved_chunks,
                conversation_history
            )
            llm_time = time.time() - llm_start
            
            total_time = time.time() - start_time
            
            logger.info(
                f"Query processed successfully. "
                f"Retrieval: {retrieval_time:.2f}s, LLM: {llm_time:.2f}s, Total: {total_time:.2f}s"
            )
            
            return {
                'response': response,
                'sources': sources,
                'retrieved_chunks': retrieved_chunks,
                'retrieval_time': retrieval_time,
                'llm_time': llm_time,
                'total_time': total_time
            }
            
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}", exc_info=True)
            return {
                'response': "I apologize, but I encountered an error while processing your question. Please try again.",
                'sources': [],
                'retrieved_chunks': [],
                'retrieval_time': 0,
                'llm_time': 0,
                'total_time': time.time() - start_time,
                'error': str(e)
            }
    
    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve relevant context chunks for a query
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve
        
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Search vector store
            k = top_k or self.config['top_k']
            results = self.vector_store.search(query_embedding, k=k)
            
            # Filter by similarity threshold
            threshold = self.config['similarity_threshold']
            filtered_results = [
                r for r in results
                if r['score'] >= threshold
            ]
            
            logger.info(
                f"Retrieved {len(filtered_results)}/{len(results)} chunks above threshold {threshold}"
            )
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in retrieve_context: {str(e)}", exc_info=True)
            return []
    
    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        conversation_history: Optional[List[Dict]] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Generate response using LLM with retrieved context
        
        Args:
            query: User's question
            retrieved_chunks: Retrieved context chunks
            conversation_history: Previous messages
        
        Returns:
            Tuple of (response_text, sources)
        """
        try:
            # Build context from chunks
            context = self._build_context(retrieved_chunks)
            
            # Build system prompt
            system_prompt = self._build_system_prompt(context)
            
            # Build messages
            messages = self._build_messages(query, conversation_history)
            
            # Generate response based on provider
            if self.llm_config['provider'] == 'gemini':
                response_text = self._generate_gemini(system_prompt, messages)
            else:
                raise ValueError(f"Unsupported provider: {self.llm_config['provider']}")
            
            # Extract sources
            sources = self._extract_sources(retrieved_chunks)
            
            return response_text, sources
            
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}", exc_info=True)
            raise
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks"""
        context_parts = []
        max_length = self.config['max_context_length']
        current_length = 0
        
        for i, chunk in enumerate(chunks, 1):
            chunk_text = f"[Source {i}: {chunk.get('document_title', 'Unknown')}]\n{chunk['content']}\n"
            
            if current_length + len(chunk_text) > max_length:
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n".join(context_parts)
    
    def _build_system_prompt(self, context: str) -> str:
        """Build system prompt with context"""
        return f"""You are a helpful customer support assistant. Your role is to answer questions accurately based on the provided context.

Guidelines:
1. Answer questions using ONLY the information from the context provided
2. If the context doesn't contain enough information, say so honestly
3. Be concise but thorough in your explanations
4. If you're not sure about something, acknowledge the uncertainty
5. Use a friendly, professional tone
6. Format your response clearly with bullet points or numbered lists when appropriate

Context Information:
{context}

Remember: Only use information from the context above. If the answer isn't in the context, politely let the user know and suggest how they might get help."""
    
    def _build_messages(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Build message list for LLM"""
        messages = []
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        # Add current query
        messages.append({
            'role': 'user',
            'content': query
        })
        
        return messages
    
    def _generate_gemini(self, system_prompt: str, messages: List[Dict]) -> str:
        """Generate response using Google Gemini"""
        try:
            # Build a simple prompt that includes system guidance and chat history
            prompt_lines = [f"System: {system_prompt}", "Conversation:"]
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                prompt_lines.append(f"{role.capitalize()}: {content}")
            prompt = "\n".join(prompt_lines)

            model = self.llm_client.GenerativeModel(self.llm_config['model'])
            response = model.generate_content(prompt)
            return getattr(response, 'text', '') or ''
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}", exc_info=True)
            raise
    
    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Extract source information from chunks"""
        sources = []
        seen_docs = set()
        
        for chunk in chunks:
            doc_id = chunk.get('document_id')
            if doc_id and doc_id not in seen_docs:
                sources.append({
                    'document_id': doc_id,
                    'document_title': chunk.get('document_title', 'Unknown'),
                    'relevance_score': chunk.get('score', 0),
                    'page': chunk.get('page_number'),
                })
                seen_docs.add(doc_id)
        
        return sources
    
    def clear_cache(self):
        """Clear any cached data"""
        logger.info("Clearing RAG engine cache")
        # Implement if you add caching
        pass
    
    def reload_index(self):
        """Reload the vector store index"""
        logger.info("Reloading vector store index")
        self.vector_store.load_index()
