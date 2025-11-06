
from typing import Dict, List, Optional
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import re

class WebSearchAgent:
    """
    Web search agent for mathematical content
    Implements MCP-style tool calling pattern
    """
    
    def __init__(self):
        self.search_client = DDGS()
        self.max_results = 3
        self.timeout = 10
        
        # Trusted educational domains for math
        self.trusted_domains = [
            'khanacademy.org',
            'mathworld.wolfram.com',
            'wikipedia.org',
            'brilliant.org',
            'mathisfun.com',
            'stackoverflow.com/questions/tagged/math',
            'math.stackexchange.com',
            'brilliant.org',
            'coursera.org',
            'mit.edu',
            'stanford.edu'
        ]
    
    def search_math_content(self, query: str) -> Dict:
        """
        Search for mathematical content using DuckDuckGo
        
        Args:
            query: Math question to search for
            
        Returns:
            Dictionary with search results and metadata
        """
        try:
            # Enhance query for better math results
            enhanced_query = self._enhance_math_query(query)
            
            # Perform search
            results = list(self.search_client.text(
                enhanced_query,
                max_results=self.max_results
            ))
            
            if not results:
                return {
                    'success': False,
                    'results': [],
                    'message': 'No search results found',
                    'source': 'web_search'
                }
            
            # Filter and rank results
            filtered_results = self._filter_results(results)
            
            # Extract content from top results
            enriched_results = []
            for result in filtered_results[:2]:  # Top 2 results
                content = self._extract_content(result.get('href', ''))
                if content:
                    enriched_results.append({
                        'title': result.get('title', ''),
                        'url': result.get('href', ''),
                        'snippet': result.get('body', ''),
                        'content': content,
                        'is_trusted': self._is_trusted_domain(result.get('href', ''))
                    })
            
            return {
                'success': True,
                'results': enriched_results,
                'total_found': len(results),
                'message': f'Found {len(enriched_results)} relevant sources',
                'source': 'web_search'
            }
            
        except Exception as e:
            return {
                'success': False,
                'results': [],
                'message': f'Search error: {str(e)}',
                'source': 'web_search'
            }
    
    def _enhance_math_query(self, query: str) -> str:
        """Add math-specific keywords to improve search results"""
        # Add context keywords if not present
        math_context = ['mathematics', 'math', 'explanation', 'how to']
        
        query_lower = query.lower()
        if not any(keyword in query_lower for keyword in math_context):
            return f"mathematics {query} explanation"
        
        return query
    
    def _filter_results(self, results: List[Dict]) -> List[Dict]:
        """Filter and rank search results by relevance and trustworthiness"""
        scored_results = []
        
        for result in results:
            score = 0
            url = result.get('href', '').lower()
            title = result.get('title', '').lower()
            body = result.get('body', '').lower()
            
            # Score trusted domains higher
            if self._is_trusted_domain(url):
                score += 10
            
            # Score educational keywords
            edu_keywords = ['tutorial', 'explanation', 'learn', 'guide', 'how to', 'step by step']
            score += sum(2 for keyword in edu_keywords if keyword in title or keyword in body)
            
            # Score math-specific content
            math_keywords = ['formula', 'equation', 'theorem', 'proof', 'solution', 'calculate']
            score += sum(1 for keyword in math_keywords if keyword in title or keyword in body)
            
            # Penalize forums/discussions (prefer authoritative sources)
            if 'forum' in url or 'reddit' in url:
                score -= 5
            
            scored_results.append((score, result))
        
        # Sort by score (descending)
        scored_results.sort(reverse=True, key=lambda x: x[0])
        
        return [result for score, result in scored_results]
    
    def _is_trusted_domain(self, url: str) -> bool:
        """Check if URL is from a trusted educational domain"""
        return any(domain in url.lower() for domain in self.trusted_domains)
    
    def _extract_content(self, url: str) -> Optional[str]:
        """
        Extract main content from a webpage
        
        Args:
            url: URL to extract content from
            
        Returns:
            Extracted text content or None
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Educational Bot)'
            }
            
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()
            
            # Get text from paragraphs
            paragraphs = soup.find_all(['p', 'div'], limit=10)
            text = ' '.join([p.get_text().strip() for p in paragraphs])
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Limit length
            if len(text) > 1000:
                text = text[:1000] + "..."
            
            return text if text else None
            
        except Exception as e:
            return None
    
    def validate_answer_exists(self, query: str, search_results: List[Dict]) -> bool:
        """
        Validate that search results actually contain relevant information
        
        Args:
            query: Original query
            search_results: Results from web search
            
        Returns:
            True if relevant information found, False otherwise
        """
        if not search_results:
            return False
        
        # Extract key terms from query
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Check if results contain query terms
        for result in search_results:
            result_text = (
                result.get('title', '') + ' ' +
                result.get('snippet', '') + ' ' +
                result.get('content', '')
            ).lower()
            
            # Count matching terms
            matches = sum(1 for term in query_terms if term in result_text)
            
            # If at least 50% of query terms found, consider valid
            if matches >= len(query_terms) * 0.5:
                return True
        
        return False
    
    def format_search_context(self, search_results: List[Dict]) -> str:
        """
        Format search results into context string for LLM
        
        Args:
            search_results: Enriched search results
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return ""
        
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            trust_badge = " [TRUSTED SOURCE]" if result.get('is_trusted') else ""
            
            context_parts.append(
                f"Source {i}{trust_badge}:\n"
                f"Title: {result.get('title', 'N/A')}\n"
                f"URL: {result.get('url', 'N/A')}\n"
                f"Content: {result.get('content', result.get('snippet', 'N/A'))}\n"
            )
        
        return "\n\n".join(context_parts)