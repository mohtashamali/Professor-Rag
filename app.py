import streamlit as st
import os
from dotenv import load_dotenv
from rag.enhanced_rag_orchestrator import EnhancedRAGOrchestrator
from pathlib import Path
import uuid

load_dotenv()

st.set_page_config(
    page_title="Professor Rag",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #000000;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .warning-message {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .error-message {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .source-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .kb-source { background-color: #d4edda; color: #155724; }
    .web-source { background-color: #d1ecf1; color: #0c5460; }
    .llm-source { background-color: #cce5ff; color: #004085; }
    .feedback-buttons {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state"""
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'pending_feedback' not in st.session_state:
        st.session_state.pending_feedback = {}
    if 'show_refinement' not in st.session_state:
        st.session_state.show_refinement = {}

def initialize_system():
    """Initialize enhanced RAG system"""
    try:
        # api_key = os.getenv("GROQ_API_KEY")
        api_key = st.secrets["GROQ_API_KEY"]
        if not api_key:
            st.error("❌ GROQ_API_KEY not found")
            return False
        
        st.session_state.orchestrator = EnhancedRAGOrchestrator(groq_api_key=api_key)
        st.session_state.system_ready = True
        return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

def clean_latex_response(text: str) -> str:
    """Clean LaTeX formatting"""
    import re
    text = re.sub(r'\$+', '', text)
    text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'##\s*Step\s+\d+:\s*([^\n]+)', r'**\1**', text)
    return text

def display_message(msg_id: int, role: str, content: str, metadata: dict = None):
    """Display chat message with feedback options"""
    content = clean_latex_response(content)
    content_html = content.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            <span style="color: #1a1a1a;">{content_html}</span>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        source = metadata.get('source', 'Unknown') if metadata else 'Unknown'
        badge_class = 'kb-source' if 'Knowledge Base' in source else ('web-source' if 'Web' in source else 'llm-source')
        source_badge = f'<span class="source-badge {badge_class}">{source}</span>'
        
        # Display sources
        sources_text = ""
        if metadata and metadata.get('sources'):
            sources_list = metadata['sources']
            if len(sources_list) <= 2:
                sources_text = f"<br><small style='color: #555;'>📄 Sources: {', '.join(sources_list)}</small>"
        
        # Validation warnings
        warning_text = ""
        if metadata:
            if metadata.get('math_relevance', 1.0) < 0.5:
                warning_text = "<br><small style='color: #ff6b6b;'>⚠️ Low math relevance detected</small>"
        
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🧮 Professor Rag:</strong> {source_badge}<br>
            <span style="color: #1a1a1a;">{content_html}</span>{sources_text}{warning_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Feedback buttons
        col1, col2, col3, col4 = st.columns([1, 1, 2, 6])
        
        with col1:
            if st.button("👍", key=f"up_{msg_id}"):
                handle_feedback(msg_id, rating=5, feedback_type="positive")
        
        with col2:
            if st.button("👎", key=f"down_{msg_id}"):
                st.session_state.show_refinement[msg_id] = True
                st.rerun()
        
        with col3:
            if st.button("🔄 Refine", key=f"refine_{msg_id}"):
                st.session_state.show_refinement[msg_id] = True
                st.rerun()
        
        # Refinement input
        if st.session_state.show_refinement.get(msg_id, False):
            st.markdown("---")
            refinement_input = st.text_area(
                "What would you like to improve?",
                key=f"refinement_text_{msg_id}",
                placeholder="E.g., 'Can you explain step 2 in more detail?' or 'Can you add an example?'"
            )
            
            col_a, col_b = st.columns([1, 5])
            with col_a:
                if st.button("Submit Refinement", key=f"submit_refine_{msg_id}"):
                    if refinement_input:
                        handle_refinement(msg_id, refinement_input)
                    else:
                        st.warning("Please provide feedback")
            
            with col_b:
                if st.button("Cancel", key=f"cancel_refine_{msg_id}"):
                    st.session_state.show_refinement[msg_id] = False
                    st.rerun()

def handle_feedback(msg_id: int, rating: int, feedback_type: str):
    """Handle user feedback"""
    if msg_id < len(st.session_state.chat_history):
        msg = st.session_state.chat_history[msg_id]
        
        if msg['role'] == 'assistant':
            # Get question (previous message)
            question = ""
            if msg_id > 0:
                question = st.session_state.chat_history[msg_id - 1]['content']
            
            # Record feedback
            feedback_id = st.session_state.orchestrator.feedback_system.record_feedback(
                question=question,
                response=msg['content'],
                source=msg.get('metadata', {}).get('source', 'Unknown'),
                rating=rating,
                feedback_text=feedback_type,
                session_id=st.session_state.session_id
            )
            
            st.success("✅ Thank you for your feedback!" if rating >= 4 else "📝 Feedback recorded")
            st.session_state.pending_feedback[msg_id] = feedback_id

def handle_refinement(msg_id: int, user_feedback: str):
    if msg_id < len(st.session_state.chat_history):
        msg = st.session_state.chat_history[msg_id]
        
        if msg['role'] == 'assistant':
            question = ""
            if msg_id > 0:
                question = st.session_state.chat_history[msg_id - 1]['content']
            
            with st.spinner("🔄 Refining response..."):
                result = st.session_state.orchestrator.refine_response(
                    original_question=question, 
                    original_response=msg['content'],
                    user_feedback=user_feedback,
                    feedback_id=st.session_state.pending_feedback.get(msg_id)
                )
            
            # Add refined response
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': result['refined_answer'],
                'metadata': {'source': 'Refined Response', 'is_refined': True}
            })
            
            st.session_state.show_refinement[msg_id] = False
            st.rerun()

def main():
    # Main application
    initialize_session_state()
    
    if not st.session_state.system_ready:
        with st.spinner("🔄 Hello,From Maths Professor..."):
            initialize_system()
    
    st.markdown('<h1 class="main-header">🧮Maths Professor</h1>', unsafe_allow_html=True)
    # st.markdown('<p style="text-align: center; color: #666;">With Guardrails, Web Search & Human Feedback</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        
        if st.session_state.system_ready:
            st.success("✅ System Active")
            
            # System health
            with st.expander("📊 System Health", expanded=False):
                health = st.session_state.orchestrator.get_system_health()
                
                st.metric("Knowledge Base", 
                         health['knowledge_base']['status'].upper(),
                         f"{health['knowledge_base']['document_count']} chunks")
                
                st.metric("Total Feedback",
                         health['feedback']['total_feedback'])
                
                st.metric("Avg Rating",
                         f"{health['feedback']['average_rating']}/5.0")

        
        st.divider()
        
        # Controls
        st.subheader("🎮 Controls")
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("📊 View Analytics"):
            stats = st.session_state.orchestrator.feedback_system.get_feedback_stats()
            st.json(stats)
        
        st.divider()
        
        # Info
        with st.expander("ℹ️ Features"):
            st.markdown("""
            **🛡️ Guardrails:**
            - Math content filtering
            - Inappropriate content blocking
            
            **📚 Knowledge Sources:**
            1. PDF Knowledge Base (Priority)
            2. Web Search (Fallback)
            3. LLM General Knowledge
            
            **🔄 Human Feedback:**
            - Rate responses (👍/👎)
            - Request refinements
            - Continuous learning
            """)
    
    # Main chat
    st.divider()
    
    # Display chat history
    for i, msg in enumerate(st.session_state.chat_history):
        display_message(i, msg['role'], msg['content'], msg.get('metadata'))
    
    # Chat input
    st.divider()
    
    if st.session_state.system_ready:
        user_question = st.chat_input("Ask a math question...")
        
        if user_question:
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_question
            })
            
            # Get response
            with st.spinner("🤔 Processing..."):
                response = st.session_state.orchestrator.answer_question(
                    question=user_question,
                    session_id=st.session_state.session_id
                )
            
            # Add assistant response
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response['answer'],
                'metadata': {
                    'source': response['source'],
                    'sources': response.get('sources', []),
                    'math_relevance': response.get('math_relevance', 1.0),
                    'used_kb': response.get('used_kb', False),
                    'used_web': response.get('used_web', False)
                }
            })
            
            st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p><strong>Created by Mohd Mohtasham ali</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()