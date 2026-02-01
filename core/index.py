"""
FAISS Index - Simplified vector search
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class VideoIndex:
    """FAISS-based index for video fingerprints"""
    
    def __init__(self, dimension: int = 2048):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension (2048 for ResNet50)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.metadata = []  # Store {content_id, timestamp, frame_idx}
    
    def add(self, embeddings: np.ndarray, content_id: str, timestamps: List[float]):
        """
        Add video fingerprints to index.
        
        Args:
            embeddings: Feature matrix (N x dimension)
            content_id: Unique identifier for this content
            timestamps: List of timestamps for each frame
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = (embeddings / norms).astype(np.float32)
        
        # Add to FAISS
        self.index.add(normalized)
        
        # Store metadata
        for idx, ts in enumerate(timestamps):
            self.metadata.append({
                'content_id': content_id,
                'timestamp': ts,
                'frame_idx': idx
            })
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for similar frames.
        
        Args:
            query_embedding: Query feature vector
            top_k: Number of results to return
        
        Returns:
            List of (metadata, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Search
        k = min(top_k, self.index.ntotal)
        similarities, indices = self.index.search(query_embedding, k)
        
        # Build results
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            # FAISS IndexFlatIP with L2-normalized vectors returns cosine similarity
            # directly in [0, 1] for non-negative features (ReLU-activated ResNet50).
            score = float(np.clip(sim, 0.0, 1.0))

            results.append((self.metadata[idx].copy(), score))
        
        return results
    
    def save(self, path: str):
        """Save index and metadata to disk"""
        path_obj = Path(path)
        path_obj.parent.mkdir(exist_ok=True, parents=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path_obj) + '.faiss')
        
        # Save metadata
        with open(str(path_obj) + '.metadata.pkl', 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'dimension': self.dimension
            }, f)
    
    def load(self, path: str):
        """Load index and metadata from disk"""
        path_obj = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path_obj) + '.faiss')
        
        # Load metadata
        with open(str(path_obj) + '.metadata.pkl', 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.dimension = data['dimension']
    
    def __len__(self):
        return self.index.ntotal
