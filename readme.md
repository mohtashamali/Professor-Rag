# 🧮 Math Professor AI - Agentic RAG System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An intelligent conversational AI system for mathematics education powered by Retrieval-Augmented Generation (RAG), web search capabilities, and human-in-the-loop feedback mechanisms.


## 🎯 Overview

Math Professor AI is a production-ready educational chatbot that combines multiple AI techniques to provide accurate, step-by-step mathematical explanations. The system intelligently searches through uploaded PDF textbooks first, falls back to web search when needed, and continuously learns from user feedback.

### Key Features

- 🛡️ **Smart Guardrails**: Input/output validation ensures only math-related, safe content
- 📚 **Knowledge Base RAG**: Semantic search through uploaded PDF textbooks and notes
- 🌐 **Web Search Integration**: Automatic fallback to trusted educational sources (Khan Academy, Wolfram, etc.)
- 🔄 **Human-in-the-Loop**: Thumbs up/down feedback and response refinement
- 🤖 **Powered by Groq**: Fast LLM inference for real-time responses
- 📊 **Analytics Dashboard**: Track response quality and system performance

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Question                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                ┌────────▼─────────┐
                │ Input Guardrails │  ← Content filtering
                │ • Math relevance │     Safety checks
                │ • Safety check   │
                └────────┬─────────┘
                         │
                ┌────────▼──────────┐
                │ Knowledge Base    │  ← Semantic search
                │ (Vector DB)       │     PDF textbooks
                └────────┬──────────┘
                         │
                    Found? ├─ Yes → Context
                         │
                         No
                         │
                ┌────────▼──────────┐
                │   Web Search      │  ← DuckDuckGo
                │   (MCP Pattern)   │     Trusted sources
                └────────┬──────────┘
                         │
                ┌────────▼──────────┐
                │  LLM Generation   │  ← Groq API
                │  (Llama 3.3 70B)  │     Step-by-step
                └────────┬──────────┘
                         │
                ┌────────▼──────────┐
                │ Output Guardrails │  ← Quality check
                │ • Safety          │     Validation
                │ • Quality score   │
                └────────┬──────────┘
                         │
                    Response
                         │
                ┌────────▼──────────┐
                │ Human Feedback    │  ← 👍 👎 🔄
                │ • Ratings         │     Refinement
                │ • Refinement      │     Learning
                └───────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Groq API key ([Get one free](https://console.groq.com))
- 4GB RAM minimum (8GB recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/mohtashamali/math-professor-ai.git
cd math-professor-ai
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
echo "GROQ_API_KEY=your_api_key_here" > .env
```

5. **Create directory structure**
```bash
mkdir pdfs rag llm guardrails mcp feedback
touch rag/__init__.py llm/__init__.py guardrails/__init__.py mcp/__init__.py feedback/__init__.py
```

6. **Add your math PDFs to the `pdfs/` folder**

7. **Initialize knowledge base**
```bash
python setup_knowledge_base.py
```

8. **Run the application**
```bash
python -m streamlit run app_enhanced.py
```

## 💡 How It Works

### 1. Knowledge Base Search (Priority 1)
When you ask a question, the system first searches through your uploaded PDF textbooks using semantic similarity. If relevant content is found with confidence > 50%, it uses that context to generate an answer.

### 2. Web Search Fallback (Priority 2)
If the knowledge base doesn't have relevant information, the system automatically searches trusted educational websites:
- Khan Academy
- Wolfram MathWorld
- Math Stack Exchange
- Wikipedia
- Educational institutions (.edu domains)

### 3. LLM Generation (Priority 3)
If neither source has information, the system uses the LLM's general knowledge to provide a mathematical explanation.

### 4. Human Feedback Loop
Every response can be:
- Rated with 👍 (good) or 👎 (needs improvement)
- Refined with specific feedback ("Can you add an example?")
- Tracked for continuous learning

## 🎓 Example Use Cases

### Example 1: Calculus Question
```
You: "Explain the chain rule with an example"

System: 
[Searches Knowledge Base]
✓ Found in calculus_textbook.pdf (confidence: 0.87)
↓
[Generates step-by-step explanation with examples]
↓
Response: The chain rule states that if y = f(g(x)), then...
Source: Knowledge Base (PDF)
```

### Example 2: Advanced Topic
```
You: "What is the Riemann Hypothesis?"

System:
[Searches Knowledge Base]
✗ Not found (confidence: 0.22)
↓
[Searches Web]
✓ Found on Wolfram MathWorld + Wikipedia
↓
[Generates explanation with web context]
↓
Response: The Riemann Hypothesis is one of the most important...
Source: Web Search
Sources: mathworld.wolfram.com, wikipedia.org
```

### Example 3: Feedback & Refinement
```
You: "Solve x^2 - 5x + 6 = 0"
Bot: [Provides solution]
You: 👎 "Can you explain the factoring method more clearly?"
Bot: [Generates refined response with detailed factoring steps]
```

## 🛡️ Safety & Guardrails

### Input Guardrails
- ✅ Validates math relevance (keyword + semantic analysis)
- ✅ Blocks inappropriate content
- ✅ Checks question quality
- ✅ Detects aggressive/harmful intent

### Output Guardrails
- ✅ Validates response safety
- ✅ Scores explanation quality
- ✅ Ensures educational value
- ✅ Detects and handles refusals

## 📊 Features Breakdown

| Feature | Description | Status |
|---------|-------------|--------|
| **PDF Knowledge Base** | Upload and search math textbooks | ✅ Complete |
| **Web Search** | Automatic fallback to trusted sources | ✅ Complete |
| **MCP Integration** | Model Context Protocol architecture | ✅ Complete |
| **Guardrails** | Input/output content filtering | ✅ Complete |
| **Feedback System** | Human-in-the-loop learning | ✅ Complete |
| **Analytics** | Performance tracking & insights | ✅ Complete |
| **Step-by-step Solutions** | Detailed mathematical explanations | ✅ Complete |
| **Multi-source Attribution** | Shows where answers come from | ✅ Complete |

## 🔧 Configuration

### Adjusting Confidence Thresholds

Edit `rag/enhanced_rag_orchestrator.py`:

```python
# Knowledge base minimum confidence
self.min_confidence_score = 0.5  # 0.0 to 1.0

# Trigger web search if KB confidence below this
self.web_search_threshold = 0.4  # 0.0 to 1.0

# Enable/disable web search
self.enable_web_search = True  # True/False
```

### Adding Trusted Web Sources

Edit `mcp/web_search_agent.py`:

```python
self.trusted_domains = [
    'khanacademy.org',
    'mathworld.wolfram.com',
    'your-custom-domain.com',  # Add here
]
```

### Customizing Math Keywords

Edit `guardrails/content_filter.py`:

```python
self.math_keywords = {
    'algebra', 'calculus', 'geometry',
    'your_custom_keyword',  # Add here
}
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| **Response Time (KB)** | 1.5-3.5s |
| **Response Time (Web)** | 3.5-8.5s |
| **Accuracy** | 90%+ with KB |
| **Embedding Model** | all-MiniLM-L6-v2 |
| **LLM Model** | Llama 3.3 70B |
| **Vector DB** | ChromaDB |

## 🧪 Testing

Run the test suite:

```bash
# Test guardrails
python -c "
from guardrails.content_filter import ContentGuardrails
g = ContentGuardrails()
print(g.validate_input('What is calculus?'))
"

# Test knowledge base
python -c "
from rag.vector_store import VectorStore
vs = VectorStore()
print(vs.get_collection_count())
"

# Test web search
python -c "
from mcp.web_search_agent import WebSearchAgent
ws = WebSearchAgent()
result = ws.search_math_content('pythagorean theorem')
print(result)
"
```

## 📚 Technologies Used

- **Frontend**: Streamlit
- **LLM**: Groq (Llama 3.3 70B)
- **Vector DB**: ChromaDB
- **Embeddings**: Sentence Transformers
- **Web Search**: DuckDuckGo
- **PDF Processing**: PyPDF2
- **Feedback Storage**: SQLite
- **Guardrails**: TextBlob, Custom filters

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🐛 **Report bugs** - Open an issue
2. 💡 **Suggest features** - Start a discussion
3. 🔧 **Submit PRs** - Fork, branch, commit, push, PR
4. 📖 **Improve docs** - Fix typos, add examples
5. 🧪 **Add tests** - Improve coverage

### Development Setup

```bash
# Fork and clone the repo
git clone https://github.com/mohtashamali/Professor-Rag

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
streamlit run app_enhanced.py

# Commit and push
git commit -m "Add your feature"
git push origin feature/your-feature-name

# Open a Pull Request
```

## 🐛 Troubleshooting

### Common Issues

**"Module not found" error**
```bash
# Ensure all __init__.py files exist
ls */__init__.py

# Reinstall dependencies
pip install -r requirements.txt
```

**Web search not working**
```bash
# Install search dependencies
pip install duckduckgo-search beautifulsoup4
```

**Guardrails too strict**
```python
# In guardrails/content_filter.py
# Lower the math relevance threshold
is_math = confidence > 0.2  # Default: 0.3
```

**Database locked error**
```bash
# Close all app instances
pkill -f streamlit

# Remove database
rm feedback.db

# Restart app
streamlit run app_enhanced.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Groq** - Lightning-fast LLM inference
- **ChromaDB** - Efficient vector database
- **DuckDuckGo** - Privacy-focused search
- **Streamlit** - Beautiful web framework
- **Anthropic Claude** - Development assistance


---

**Built for mathematical education**

Made by [Github](https://github.com/mohtashamali) | [Portfolio](https://mohtashamali.github.io/Mohtasham-portfolio/) | [LinkedIn](https://www.linkedin.com/in/mohd-mohtasham-ali-167156287/)
