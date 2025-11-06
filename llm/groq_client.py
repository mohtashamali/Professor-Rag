"""
Groq LLM Client Module
Handles all LLM interactions using Groq API
"""

import os
from groq import Groq
from typing import Optional

class GroqClient:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq client with API key"""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not found. Please set GROQ_API_KEY environment variable.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"  # Updated model (Nov 2024)
    
    def generate_response(self, question: str, context: Optional[str] = None) -> str:
        """
        Generate a response using Groq LLM
        
        Args:
            question: User's question
            context: Optional context from PDF knowledge base
            
        Returns:
            Generated response as string
        """
        try:
            # Create system prompt for math professor persona
            system_prompt = """You are an expert Mathematics Professor with deep knowledge in:
- Calculus (differential and integral)
- Linear Algebra
- Differential Equations
- Real and Complex Analysis
- Probability and Statistics
- Abstract Algebra
- Topology
- Number Theory

When answering questions:
1. Provide step-by-step solutions
2. Explain the reasoning behind each step
3. Use plain text for mathematical expressions (avoid LaTeX commands like \boxed, \text, \mathbb, etc.)
4. Write equations naturally: use "x^2" instead of "x²", "sqrt(x)" instead of "√x"
5. Include examples when helpful
6. Break down complex concepts into understandable parts
7. Be precise and rigorous in mathematical explanations
8. Keep responses conversational and natural
9. Do NOT use $ signs, \boxed{}, or other LaTeX formatting
10. Write mathematics in plain readable text

IMPORTANT: Write all mathematical content in plain text format that's easy to read in a chat interface."""

            # Build user message
            if context:
                user_message = f"""Based on the following context from my knowledge base:

{context}

Please answer this question: {question}

Provide a detailed, step-by-step explanation."""
            else:
                user_message = f"""I couldn't find relevant information in my knowledge base for this question.

Question: {question}

Please provide a comprehensive, step-by-step explanation using your mathematical expertise."""

            # Call Groq API
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=self.model,
                temperature=0.3,  # Lower temperature for more focused, accurate responses
                max_tokens=2048,
                top_p=0.9
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response from LLM: {str(e)}"
    
    def generate_followup_response(self, question: str, chat_history: list) -> str:
        """
        Generate response considering chat history for follow-up questions
        
        Args:
            question: Current user question
            chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            Generated response as string
        """
        try:
            system_prompt = """You are an expert Mathematics Professor. Maintain context from the conversation and provide clear, step-by-step mathematical explanations."""
            
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(chat_history[-6:])  # Include last 3 exchanges for context
            messages.append({"role": "user", "content": question})
            
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=2048,
                top_p=0.9
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"