"""
Human-in-the-Loop Feedback System
Collects user feedback and enables response refinement
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class FeedbackSystem:
    """Manages user feedback and response refinement"""
    
    def __init__(self, db_path: str = "./feedback.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Create feedback database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                response TEXT NOT NULL,
                source TEXT NOT NULL,
                rating INTEGER,
                feedback_text TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                is_refined BOOLEAN DEFAULT 0
            )
        """)
        
        # Refinement history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS refinements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_feedback_id INTEGER,
                refined_response TEXT NOT NULL,
                refinement_reason TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (original_feedback_id) REFERENCES feedback (id)
            )
        """)
        
        # Analytics table for learning
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_type TEXT,
                avg_rating REAL,
                total_responses INTEGER,
                positive_feedback INTEGER,
                negative_feedback INTEGER,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_feedback(
        self,
        question: str,
        response: str,
        source: str,
        rating: Optional[int] = None,
        feedback_text: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> int:
        """
        Record user feedback for a response
        
        Args:
            question: User's original question
            response: System's response
            source: Response source (Knowledge Base/LLM/Web Search)
            rating: User rating (1-5, or thumbs up/down as 1/0)
            feedback_text: Optional text feedback
            session_id: Session identifier
            
        Returns:
            Feedback ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feedback (question, response, source, rating, feedback_text, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (question, response, source, rating, feedback_text, session_id))
        
        feedback_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        # Update analytics
        self._update_analytics(source, rating)
        
        return feedback_id
    
    def request_refinement(
        self,
        feedback_id: int,
        user_input: str
    ) -> Dict:
        """
        Mark response for refinement based on user feedback
        
        Args:
            feedback_id: ID of the original feedback
            user_input: User's refinement request/clarification
            
        Returns:
            Dictionary with refinement info
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get original feedback
        cursor.execute("""
            SELECT question, response, source FROM feedback WHERE id = ?
        """, (feedback_id,))
        
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return {'success': False, 'message': 'Feedback not found'}
        
        question, original_response, source = result
        
        # Mark as needing refinement
        cursor.execute("""
            UPDATE feedback SET is_refined = 1 WHERE id = ?
        """, (feedback_id,))
        
        conn.commit()
        conn.close()
        
        return {
            'success': True,
            'feedback_id': feedback_id,
            'question': question,
            'original_response': original_response,
            'user_input': user_input,
            'source': source
        }
    
    def store_refined_response(
        self,
        feedback_id: int,
        refined_response: str,
        refinement_reason: str
    ):
        """
        Store a refined response after human feedback
        
        Args:
            feedback_id: ID of original feedback
            refined_response: New improved response
            refinement_reason: Why it was refined
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO refinements (original_feedback_id, refined_response, refinement_reason)
            VALUES (?, ?, ?)
        """, (feedback_id, refined_response, refinement_reason))
        
        conn.commit()
        conn.close()
    
    def get_feedback_stats(self) -> Dict:
        """Get overall feedback statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total feedback count
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_feedback = cursor.fetchone()[0]
        
        # Average rating
        cursor.execute("SELECT AVG(rating) FROM feedback WHERE rating IS NOT NULL")
        avg_rating = cursor.fetchone()[0] or 0
        
        # Source breakdown
        cursor.execute("""
            SELECT source, COUNT(*) as count, AVG(rating) as avg_rating
            FROM feedback
            WHERE rating IS NOT NULL
            GROUP BY source
        """)
        source_stats = cursor.fetchall()
        
        # Recent negative feedback
        cursor.execute("""
            SELECT question, response, feedback_text
            FROM feedback
            WHERE rating IS NOT NULL AND rating <= 2
            ORDER BY timestamp DESC
            LIMIT 5
        """)
        negative_feedback = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_feedback': total_feedback,
            'average_rating': round(avg_rating, 2),
            'source_stats': [
                {'source': s[0], 'count': s[1], 'avg_rating': round(s[2], 2)}
                for s in source_stats
            ],
            'recent_negative': [
                {'question': n[0], 'response': n[1], 'feedback': n[2]}
                for n in negative_feedback
            ]
        }
    
    def get_learning_insights(self) -> Dict:
        """
        Extract insights from feedback for system improvement
        
        Returns:
            Dictionary with actionable insights
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get low-rated questions for improvement
        cursor.execute("""
            SELECT question, AVG(rating) as avg_rating, COUNT(*) as count
            FROM feedback
            WHERE rating IS NOT NULL
            GROUP BY question
            HAVING avg_rating < 3 AND count >= 2
            ORDER BY count DESC
            LIMIT 10
        """)
        
        problem_areas = cursor.fetchall()
        
        # Get high-performing responses
        cursor.execute("""
            SELECT source, COUNT(*) as count
            FROM feedback
            WHERE rating >= 4
            GROUP BY source
            ORDER BY count DESC
        """)
        
        best_sources = cursor.fetchall()
        
        conn.close()
        
        return {
            'problem_questions': [
                {'question': p[0], 'avg_rating': round(p[1], 2), 'occurrences': p[2]}
                for p in problem_areas
            ],
            'best_performing_sources': [
                {'source': b[0], 'positive_count': b[1]}
                for b in best_sources
            ]
        }
    
    def _update_analytics(self, source: str, rating: Optional[int]):
        """Update analytics based on new feedback"""
        if rating is None:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if analytics entry exists
        cursor.execute("""
            SELECT id, avg_rating, total_responses, positive_feedback, negative_feedback
            FROM response_analytics
            WHERE question_type = ?
        """, (source,))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing
            aid, avg_rating, total, positive, negative = result
            new_total = total + 1
            new_avg = ((avg_rating * total) + rating) / new_total
            new_positive = positive + (1 if rating >= 4 else 0)
            new_negative = negative + (1 if rating <= 2 else 0)
            
            cursor.execute("""
                UPDATE response_analytics
                SET avg_rating = ?, total_responses = ?, positive_feedback = ?, 
                    negative_feedback = ?, last_updated = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (new_avg, new_total, new_positive, new_negative, aid))
        else:
            # Create new
            cursor.execute("""
                INSERT INTO response_analytics 
                (question_type, avg_rating, total_responses, positive_feedback, negative_feedback)
                VALUES (?, ?, 1, ?, ?)
            """, (source, rating, 1 if rating >= 4 else 0, 1 if rating <= 2 else 0))
        
        conn.commit()
        conn.close()
    
    def export_feedback_data(self, output_path: str = "./feedback_export.json"):
        """Export all feedback data for analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM feedback")
        feedback_data = cursor.fetchall()
        
        cursor.execute("SELECT * FROM refinements")
        refinement_data = cursor.fetchall()
        
        conn.close()
        
        export_data = {
            'feedback': feedback_data,
            'refinements': refinement_data,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return output_path