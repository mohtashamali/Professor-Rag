"""
Vector Store Module
Handles document embedding and similarity search using ChromaDB
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import os

class VectorStore:
    def __init__(self, collection_name: str = "math_knowledge", persist_directory: str = "./chroma_db"):
        """
        Initialize Vector Store with ChromaDB and sentence transformers
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize embedding model (using a good math-capable model)
        # print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            # print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def add_documents(self, chunks: List[Dict]) -> None:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
        """
        if not chunks:
            print("No chunks to add")
            return
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        # Prepare data for ChromaDB
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = [self.embed_text(text) for text in texts]
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            self.collection.add(
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        print(f"Successfully added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, n_results: int = 3, score_threshold: float = 0.5) -> List[Dict]:
        """
        Search for relevant documents using semantic similarity
        
        Args:
            query: Search query
            n_results: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        # Generate query embedding
        query_embedding = self.embed_text(query)
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):

                distance = results['distances'][0][i]
                similarity_score = 1 - distance
                
                # Only include results above threshold
                if similarity_score >= score_threshold:
                    formatted_results.append({
                        'text': doc,
                        'metadata': results['metadatas'][0][i],
                        'score': similarity_score
                    })
        
        return formatted_results
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection"""
        try:
            return self.collection.count()
        except:
            return 0