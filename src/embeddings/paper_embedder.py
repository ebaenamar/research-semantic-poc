"""
Paper Embedder Module
Generates semantic embeddings from paper abstracts and metadata
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import json


class PaperEmbedder:
    """
    Generate semantic embeddings for research papers
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedder with specified model
        
        Args:
            model_name: HuggingFace model name for embeddings
                       Default: all-MiniLM-L6-v2 (fast, good quality)
                       Alternative: allenai/specter (specialized for papers)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
    def prepare_text(self, row: pd.Series) -> str:
        """
        Prepare text for embedding by combining relevant fields
        
        Args:
            row: DataFrame row with paper metadata
            
        Returns:
            Combined text string
        """
        parts = []
        
        # Title (most important)
        if pd.notna(row.get('title')):
            parts.append(f"Title: {row['title']}")
            
        # Abstract (core content)
        if pd.notna(row.get('abstract')):
            parts.append(f"Abstract: {row['abstract']}")
            
        # Keywords if available
        if pd.notna(row.get('keywords')):
            parts.append(f"Keywords: {row['keywords']}")
            
        # MeSH terms if available (medical subject headings)
        if pd.notna(row.get('mesh_terms')):
            parts.append(f"MeSH: {row['mesh_terms']}")
            
        return " ".join(parts)
    
    def embed_papers(
        self, 
        df: pd.DataFrame,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for all papers in dataframe
        
        Args:
            df: DataFrame with paper data
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings (n_papers x embedding_dim)
        """
        print(f"Preparing text for {len(df)} papers...")
        texts = [self.prepare_text(row) for _, row in df.iterrows()]
        
        # Filter out empty texts
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        
        print(f"Generating embeddings for {len(valid_texts)} valid papers...")
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Create full embedding array with NaN for invalid entries
        full_embeddings = np.full((len(df), embeddings.shape[1]), np.nan)
        full_embeddings[valid_indices] = embeddings
        
        return full_embeddings
    
    def save_embeddings(
        self, 
        embeddings: np.ndarray, 
        filepath: str,
        metadata: Optional[Dict] = None
    ):
        """Save embeddings and metadata to file"""
        data = {
            'embeddings': embeddings,
            'model_name': self.model_name,
            'metadata': metadata or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Embeddings saved to {filepath}")
    
    @staticmethod
    def load_embeddings(filepath: str) -> Dict:
        """Load embeddings from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def compute_similarity(
        self, 
        embeddings: np.ndarray,
        query_idx: int,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Find most similar papers to a query paper
        
        Args:
            embeddings: Array of all embeddings
            query_idx: Index of query paper
            top_k: Number of similar papers to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_embedding = embeddings[query_idx:query_idx+1]
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Get top k (excluding self)
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        return [(idx, similarities[idx]) for idx in top_indices]


if __name__ == "__main__":
    # Example usage
    print("Paper Embedder Module")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    from paper_embedder import PaperEmbedder
    import pandas as pd
    
    # Load data
    df = pd.read_csv('data/aiscientist/pubmed_data_2000.csv')
    
    # Create embedder
    embedder = PaperEmbedder()
    
    # Generate embeddings
    embeddings = embedder.embed_papers(df)
    
    # Save embeddings
    embedder.save_embeddings(
        embeddings, 
        'output/embeddings.pkl',
        metadata={'n_papers': len(df)}
    )
    """)
