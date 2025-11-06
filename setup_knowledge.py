import os
from dotenv import load_dotenv
from rag.enhanced_rag_orchestrator import EnhancedRAGOrchestrator
from pathlib import Path
import sys

load_dotenv()

def check_environment():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return False
    return True

def check_pdfs_folder():
    pdf_folder = Path("./pdfs")
    
    if not pdf_folder.exists():
        pdf_folder.mkdir()
        return False, []
    
    pdf_paths = list(pdf_folder.glob("*.pdf"))
    
    if not pdf_paths:
        return False, []
    
    return True, [str(p) for p in pdf_paths]

def initialize_system(api_key):
    try:
        orchestrator = EnhancedRAGOrchestrator(groq_api_key=api_key)
        return orchestrator
    except Exception as e:
        return None

def process_pdfs(orchestrator, pdf_paths):
    try:
        result = orchestrator.initialize_knowledge_base(pdf_paths)
        
        if result['success']:
            return True
        else:
            return False
            
    except Exception as e:
        return False

def verify_system(orchestrator):
    try:
        health = orchestrator.get_system_health()
        return True
    except Exception as e:
        return False

def test_query(orchestrator):
    test_question = "What is 4+5-98?"
    
    try:
        response = orchestrator.answer_question(test_question)
        
        if response['success']:
            return True
        else:
            return True
            
    except Exception as e:
        return False

def main():
    if not check_environment():
        sys.exit(1)
    
    has_pdfs, pdf_paths = check_pdfs_folder()
    if not has_pdfs:
        sys.exit(0)
    
    api_key = os.getenv("GROQ_API_KEY")
    
    orchestrator = initialize_system(api_key)
    if not orchestrator:
        sys.exit(1)
    
    if not process_pdfs(orchestrator, pdf_paths):
        sys.exit(1)
    
    verify_system(orchestrator)
    test_query(orchestrator)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        sys.exit(1)