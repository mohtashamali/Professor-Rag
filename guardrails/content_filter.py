

import re
from typing import Dict, Tuple
from textblob import TextBlob

class ContentGuardrails:
    """Guardrails to ensure only math education content is processed"""
    
    def __init__(self):
        # Mathematical keywords and topics
        self.math_keywords = {
            'algebra', 'calculus', 'geometry', 'trigonometry', 'statistics',
            'probability', 'equation', 'derivative', 'integral', 'matrix',
            'vector', 'function', 'theorem', 'proof', 'solve', 'calculate',
            'graph', 'formula', 'number', 'polynomial', 'exponential',
            'logarithm', 'differential', 'limit', 'series', 'sequence',
            'angle', 'triangle', 'circle', 'sine', 'cosine', 'tangent',
            'mean', 'median', 'variance', 'distribution', 'regression',
            'topology', 'analysis', 'linear', 'optimization', 'pi', 'ratio'
        }
        
        # Forbidden content categories
        self.forbidden_keywords = {
            'violence', 'weapon', 'hate', 'explicit', 'illegal', 'drug',
            'nsfw', 'adult', 'harmful', 'suicide', 'bomb', 'kill'
        }
        
        # Educational math topics (comprehensive)
        self.math_topics = {
            'arithmetic', 'algebra', 'geometry', 'trigonometry', 'calculus',
            'linear_algebra', 'differential_equations', 'statistics',
            'probability', 'number_theory', 'discrete_math', 'topology',
            'real_analysis', 'complex_analysis', 'abstract_algebra',
            'combinatorics', 'graph_theory', 'set_theory', 'logic'
        }
    
    def validate_input(self, user_input: str) -> Dict:
        """
        Validate user input for math education relevance
        
        Args:
            user_input: User's question/input
            
        Returns:
            Dictionary with validation results
        """
        input_lower = user_input.lower()
        
        # Check 1: Forbidden content
        for keyword in self.forbidden_keywords:
            if keyword in input_lower:
                return {
                    'is_valid': False,
                    'reason': 'inappropriate_content',
                    'message': 'I can only help with mathematics education. Please ask a math-related question.',
                    'severity': 'high'
                }
        
        # Check 2: Math relevance
        math_score = sum(1 for keyword in self.math_keywords if keyword in input_lower)
        
        # Check 3: Question quality (too short/vague)
        if len(user_input.strip()) < 5:
            return {
                'is_valid': False,
                'reason': 'too_short',
                'message': 'Please provide a more detailed question.',
                'severity': 'low'
            }
        
        # Check 4: Sentiment analysis (detect aggressive/inappropriate tone)
        try:
            blob = TextBlob(user_input)
            sentiment = blob.sentiment.polarity
            
            if sentiment < -0.5:  # Very negative
                return {
                    'is_valid': False,
                    'reason': 'negative_tone',
                    'message': 'Please rephrase your question in a respectful manner.',
                    'severity': 'medium'
                }
        except:
            pass  # If sentiment analysis fails, continue
        
        # Check 5: Math relevance threshold
        if math_score == 0 and not self._contains_numbers_or_symbols(user_input):
            return {
                'is_valid': True,  # Allow but flag
                'reason': 'low_math_relevance',
                'message': 'This doesn\'t seem to be a math question. I\'m optimized for mathematics education.',
                'severity': 'low',
                'warning': True
            }
        
        # All checks passed
        return {
            'is_valid': True,
            'reason': 'valid',
            'message': 'Input validated successfully',
            'severity': 'none',
            'math_score': math_score
        }
    
    def validate_output(self, response: str) -> Dict:
        """
        Validate generated response for quality and appropriateness
        
        Args:
            response: Generated response from LLM
            
        Returns:
            Dictionary with validation results
        """
        response_lower = response.lower()
        
        # Check 1: Response length (too short = poor quality)
        if len(response.strip()) < 20:
            return {
                'is_valid': False,
                'reason': 'response_too_short',
                'message': 'Generated response is too short. Regenerating...',
                'severity': 'medium'
            }
        
        # Check 2: Check for harmful content in response
        for keyword in self.forbidden_keywords:
            if keyword in response_lower:
                return {
                    'is_valid': False,
                    'reason': 'inappropriate_output',
                    'message': 'Response contains inappropriate content.',
                    'severity': 'high'
                }
        
        # Check 3: Check for "I don't know" or refusal patterns
        refusal_patterns = [
            "i don't know", "i cannot", "i'm not sure", "i don't have",
            "cannot determine", "insufficient information"
        ]
        
        has_refusal = any(pattern in response_lower for pattern in refusal_patterns)
        
        # Check 4: Educational quality - should have explanations
        has_explanation = any(word in response_lower for word in [
            'because', 'therefore', 'thus', 'hence', 'since', 'which means',
            'step', 'first', 'second', 'next', 'finally', 'explanation'
        ])
        
        quality_score = 0
        if has_explanation:
            quality_score += 1
        if len(response) > 100:
            quality_score += 1
        if not has_refusal:
            quality_score += 1
        
        return {
            'is_valid': True,
            'reason': 'valid',
            'message': 'Output validated successfully',
            'severity': 'none',
            'quality_score': quality_score,
            'has_refusal': has_refusal,
            'has_explanation': has_explanation
        }
    
    def _contains_numbers_or_symbols(self, text: str) -> bool:
        """Check if text contains mathematical numbers or symbols"""
        math_patterns = [
            r'\d+',  # Numbers
            r'[+\-*/=<>≤≥≠∞∑∏∫√π]',  # Math symbols
            r'x|y|z|n|f\(|g\(',  # Common variables
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def is_math_related(self, text: str) -> Tuple[bool, float]:
        """
        Determine if text is mathematics-related with confidence score
        
        Returns:
            Tuple of (is_math_related: bool, confidence: float)
        """
        text_lower = text.lower()
        
        # Count math keywords
        keyword_count = sum(1 for keyword in self.math_keywords if keyword in text_lower)
        
        # Check for numbers and symbols
        has_numbers = self._contains_numbers_or_symbols(text)
        
        # Calculate confidence
        confidence = 0.0
        
        if keyword_count > 0:
            confidence += min(keyword_count * 0.2, 0.6)  # Max 0.6 from keywords
        
        if has_numbers:
            confidence += 0.3
        
        # Check for question patterns
        question_words = ['what', 'how', 'why', 'when', 'where', 'solve', 'find', 'calculate', 'prove']
        if any(word in text_lower for word in question_words):
            confidence += 0.1
        
        is_math = confidence > 0.3  # Threshold for math-related
        
        return is_math, min(confidence, 1.0)