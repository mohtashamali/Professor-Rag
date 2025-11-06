"""
Enhanced RAG Orchestrator with Guardrails, Web Search, and Feedback
Implements complete agentic workflow with MCP-style architecture
"""

from typing import Dict, Optional
from rag.pdf_processor import PDFProcessor
from rag.vector_store import VectorStore
from llm.groq_client import GroqClient
from guardrails.content_filter import ContentGuardrails
from mcp.web_search_agent import WebSearchAgent
from feedback.human_loop import FeedbackSystem

class EnhancedRAGOrchestrator:
    
    def __init__(self, groq_api_key: str):
        """Initialize all components"""
        # Core RAG components
        self.pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
        self.vector_store = VectorStore()
        self.llm_client = GroqClient(api_key=groq_api_key)
        
        # Enhanced components
        self.guardrails = ContentGuardrails()
        self.web_search = WebSearchAgent()
        self.feedback_system = FeedbackSystem()
        
        # Configuration
        self.min_confidence_score = 0.5
        self.enable_web_search = True
        self.web_search_threshold = 0.4  # If KB confidence < this, try web search
    
    def initialize_knowledge_base(self, pdf_paths: list) -> Dict:
        """Load PDFs into knowledge base"""
        try:
            chunks = self.pdf_processor.process_pdfs(pdf_paths)
            
            if not chunks:
                return {
                    'success': False,
                    'message': 'No content extracted from PDFs',
                    'chunks_count': 0
                }
            
            self.vector_store.clear_collection()
            self.vector_store.add_documents(chunks)
            
            return {
                'success': True,
                'message': f'Successfully loaded {len(pdf_paths)} PDFs',
                'chunks_count': len(chunks),
                'pdfs_loaded': len(pdf_paths)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'chunks_count': 0
            }
    
    def answer_question(
        self,
        question: str,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Main pipeline: Guardrails → KB Search → Web Search → Response → Validation
        
        Args:
            question: User's question
            session_id: Optional session identifier for feedback tracking
            
        Returns:
            Complete response with metadata
        """
        # STEP 1: INPUT GUARDRAILS
        # print("🛡️ Step 1: Validating input...")
        input_validation = self.guardrails.validate_input(question)
        
        if not input_validation['is_valid'] and input_validation['severity'] == 'high':
            return {
                'answer': input_validation['message'],
                'source': 'Guardrails Blocked',
                'validation': input_validation,
                'success': False
            }
        
        # Check math relevance
        is_math, confidence = self.guardrails.is_math_related(question)
        
        if not is_math and confidence < 0.3:
            return {
                'answer': "I'm a mathematics education assistant. Please ask a math-related question, and I'll be happy to help!",
                'source': 'Math Filter',
                'validation': input_validation,
                'success': False,
                'math_confidence': confidence
            }
        
        # STEP 2: KNOWLEDGE BASE SEARCH
        # print("📚 Step 2: Searching knowledge base...")
        kb_results = self.vector_store.search(
            query=question,
            n_results=3,
            score_threshold=self.min_confidence_score
        )
        
        has_kb_context = len(kb_results) > 0
        best_kb_score = kb_results[0]['score'] if kb_results else 0
        
        # STEP 3: WEB SEARCH 
        web_context = None
        web_results = None
        
        if self.enable_web_search and (not has_kb_context or best_kb_score < self.web_search_threshold):
            # print("🌐 Step 3: Knowledge base insufficient, searching web...")
            web_search_result = self.web_search.search_math_content(question)
            
            if web_search_result['success'] and web_search_result['results']:
                web_results = web_search_result['results']
                
                # Validate that web results actually answer the question
                if self.web_search.validate_answer_exists(question, web_results):
                    web_context = self.web_search.format_search_context(web_results)
                    print(f"✅ Found {len(web_results)} relevant web sources")
                else:
                    print("⚠️ Web results not relevant enough")
                    web_results = None
        
        # STEP 4: PREPARE CONTEXT
        final_context = None
        sources = []
        source_type = "LLM General Knowledge"
        
        if has_kb_context:
            # Use knowledge base
            context_parts = []
            for i, chunk in enumerate(kb_results):
                source = chunk['metadata'].get('source', 'Unknown')
                score = chunk['score']
                context_parts.append(
                    f"[PDF Source {i+1}: {source} (Relevance: {score:.2f})]\n{chunk['text']}"
                )
                if source not in sources:
                    sources.append(source)
            
            final_context = "\n\n".join(context_parts)
            source_type = "Knowledge Base (PDF)"
            
        elif web_context:
            # Use web search results
            final_context = web_context
            source_type = "Web Search"
            sources = [r['url'] for r in web_results[:2]]
        
        # STEP 5: GENERATE RESPONSE
        # print("🤖 Step 5: Generating response...")
        answer = self.llm_client.generate_response(question, final_context)
        
        # STEP 6: OUTPUT GUARDRAILS
        # print("🛡️ Step 6: Validating output...")
        output_validation = self.guardrails.validate_output(answer)
        
        if not output_validation['is_valid'] and output_validation['severity'] == 'high':
            # Regenerate if output failed validation
            # print("⚠️ Output failed validation, regenerating...")
            answer = self.llm_client.generate_response(
                question + " [Please provide a safe, educational response]",
                final_context
            )
            output_validation = self.guardrails.validate_output(answer)
        
        # STEP 7: PREPARE RESPONSE
        response = {
            'answer': answer,
            'source': source_type,
            'sources': sources,
            'used_kb': has_kb_context,
            'used_web': web_results is not None,
            'kb_confidence': best_kb_score,
            'math_relevance': confidence,
            'input_validation': input_validation,
            'output_validation': output_validation,
            'success': True,
            'session_id': session_id
        }
        
        return response
    
    def refine_response(
        self,
        original_question: str,
        original_response: str,
        user_feedback: str,
        feedback_id: Optional[int] = None
    ) -> Dict:
        """
        Refine response based on human feedback
        
        Args:
            original_question: Original question
            original_response: Original answer
            user_feedback: User's feedback/clarification request
            feedback_id: Optional feedback ID for tracking
            
        Returns:
            Refined response
        """
        print("🔄 Refining response based on user feedback...")
        
        # Create refinement prompt
        refinement_prompt = f"""
Original Question: {original_question}

Original Answer: {original_response}

User Feedback: {user_feedback}

Please provide an improved answer that addresses the user's feedback.
Make sure to:
1. Address the specific concerns mentioned
2. Provide more clarity where requested
3. Include additional examples if needed
4. Maintain mathematical accuracy
"""
        
        # Generate refined response
        refined_answer = self.llm_client.generate_response(refinement_prompt, None)
        
        # Validate refined response
        validation = self.guardrails.validate_output(refined_answer)
        
        # Store refinement in feedback system
        if feedback_id:
            self.feedback_system.store_refined_response(
                feedback_id=feedback_id,
                refined_response=refined_answer,
                refinement_reason=user_feedback
            )
        
        return {
            'refined_answer': refined_answer,
            'validation': validation,
            'success': True
        }
    
    def get_system_health(self) -> Dict:
        """Get comprehensive system health metrics"""
        kb_status = self.vector_store.get_collection_count()
        feedback_stats = self.feedback_system.get_feedback_stats()
        learning_insights = self.feedback_system.get_learning_insights()
        
        return {
            'knowledge_base': {
                'status': 'active' if kb_status > 0 else 'empty',
                'document_count': kb_status
            },
            'feedback': feedback_stats,
            'learning_insights': learning_insights,
            'components': {
                'guardrails': 'active',
                'web_search': 'active' if self.enable_web_search else 'disabled',
                'feedback_system': 'active'
            }
        }