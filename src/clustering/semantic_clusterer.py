"""
Semantic Clusterer Module
Clusters papers based on semantic embeddings
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import hdbscan
import umap


class SemanticClusterer:
    """
    Cluster research papers based on semantic similarity
    """
    
    def __init__(self, method: str = 'hdbscan'):
        """
        Initialize clusterer
        
        Args:
            method: Clustering method ('hdbscan', 'dbscan', 'hierarchical')
        """
        self.method = method
        self.clusters = None
        self.cluster_labels = None
        self.reduced_embeddings = None
        
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_components: int = 5,
        n_neighbors: int = 15,
        min_dist: float = 0.1
    ) -> np.ndarray:
        """
        Reduce embedding dimensions using UMAP
        
        Args:
            embeddings: High-dimensional embeddings
            n_components: Target dimensionality
            n_neighbors: UMAP parameter
            min_dist: UMAP parameter
            
        Returns:
            Reduced embeddings
        """
        print(f"Reducing dimensions from {embeddings.shape[1]} to {n_components}...")
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',
            random_state=42
        )
        
        reduced = reducer.fit_transform(embeddings)
        self.reduced_embeddings = reduced
        
        return reduced
    
    def cluster_hdbscan(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 10,
        min_samples: int = 5
    ) -> np.ndarray:
        """
        Cluster using HDBSCAN (density-based, handles noise well)
        
        Args:
            embeddings: Paper embeddings
            min_cluster_size: Minimum cluster size
            min_samples: HDBSCAN parameter
            
        Returns:
            Cluster labels (-1 for noise)
        """
        print("Clustering with HDBSCAN...")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"Found {n_clusters} clusters")
        print(f"Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
        
        return labels
    
    def cluster_dbscan(
        self,
        embeddings: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> np.ndarray:
        """Cluster using DBSCAN"""
        print("Clustering with DBSCAN...")
        
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clusterer.fit_predict(embeddings)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Found {n_clusters} clusters")
        
        return labels
    
    def cluster_hierarchical(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 10
    ) -> np.ndarray:
        """Cluster using hierarchical clustering"""
        print(f"Clustering with hierarchical (n={n_clusters})...")
        
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity='cosine',
            linkage='average'
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        return labels
    
    def cluster(
        self,
        embeddings: np.ndarray,
        reduce_dims: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Main clustering method
        
        Args:
            embeddings: Paper embeddings
            reduce_dims: Whether to reduce dimensions first
            **kwargs: Method-specific parameters
            
        Returns:
            Cluster labels
        """
        # Handle NaN values
        valid_mask = ~np.isnan(embeddings).any(axis=1)
        valid_embeddings = embeddings[valid_mask]
        
        print(f"Clustering {len(valid_embeddings)} valid embeddings...")
        
        # Optionally reduce dimensions
        if reduce_dims:
            valid_embeddings = self.reduce_dimensions(valid_embeddings, **kwargs)
        
        # Cluster based on method
        if self.method == 'hdbscan':
            labels = self.cluster_hdbscan(valid_embeddings, **kwargs)
        elif self.method == 'dbscan':
            labels = self.cluster_dbscan(valid_embeddings, **kwargs)
        elif self.method == 'hierarchical':
            labels = self.cluster_hierarchical(valid_embeddings, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Map back to full array
        full_labels = np.full(len(embeddings), -1, dtype=int)
        full_labels[valid_mask] = labels
        
        self.cluster_labels = full_labels
        
        return full_labels
    
    def get_cluster_stats(
        self,
        df: pd.DataFrame,
        labels: np.ndarray
    ) -> Dict:
        """
        Get statistics for each cluster
        
        Args:
            df: DataFrame with paper data
            labels: Cluster labels
            
        Returns:
            Dictionary with cluster statistics
        """
        stats = {}
        
        unique_labels = sorted(set(labels))
        
        for label in unique_labels:
            if label == -1:
                name = "noise"
            else:
                name = f"cluster_{label}"
            
            mask = labels == label
            cluster_df = df[mask]
            
            stats[name] = {
                'size': int(mask.sum()),
                'percentage': float(mask.sum() / len(df) * 100),
                'papers': cluster_df.index.tolist(),
                'sample_titles': cluster_df['title'].head(5).tolist() if 'title' in cluster_df else []
            }
        
        return stats
    
    def extract_cluster_characteristics(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        cluster_id: int
    ) -> Dict:
        """
        Extract key characteristics of a cluster
        
        Args:
            df: DataFrame with paper data
            labels: Cluster labels
            cluster_id: ID of cluster to analyze
            
        Returns:
            Dictionary with cluster characteristics
        """
        mask = labels == cluster_id
        cluster_df = df[mask]
        
        characteristics = {
            'cluster_id': cluster_id,
            'size': int(mask.sum()),
            'papers': []
        }
        
        # Extract paper info
        for idx, row in cluster_df.iterrows():
            paper = {
                'index': int(idx),
                'title': row.get('title', ''),
                'abstract': row.get('abstract', '')[:200] + '...' if pd.notna(row.get('abstract')) else '',
                'year': row.get('year', ''),
                'citations': row.get('citations', 0)
            }
            characteristics['papers'].append(paper)
        
        # TODO: Add more sophisticated analysis:
        # - Common methodologies (extract from abstracts)
        # - Common keywords/MeSH terms
        # - Temporal trends
        # - Citation patterns
        
        return characteristics
    
    def save_clusters(
        self,
        filepath: str,
        df: pd.DataFrame,
        labels: np.ndarray
    ):
        """Save cluster analysis to JSON"""
        stats = self.get_cluster_stats(df, labels)
        
        # Get characteristics for each cluster
        unique_labels = [l for l in sorted(set(labels)) if l != -1]
        characteristics = {}
        
        for label in unique_labels:
            characteristics[f"cluster_{label}"] = self.extract_cluster_characteristics(
                df, labels, label
            )
        
        output = {
            'summary': stats,
            'characteristics': characteristics,
            'method': self.method
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Cluster analysis saved to {filepath}")


if __name__ == "__main__":
    print("Semantic Clusterer Module")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    from semantic_clusterer import SemanticClusterer
    from paper_embedder import PaperEmbedder
    import pandas as pd
    
    # Load embeddings
    embeddings = PaperEmbedder.load_embeddings('output/embeddings.pkl')
    df = pd.read_csv('data/aiscientist/pubmed_data_2000.csv')
    
    # Cluster
    clusterer = SemanticClusterer(method='hdbscan')
    labels = clusterer.cluster(embeddings['embeddings'])
    
    # Save analysis
    clusterer.save_clusters('output/clusters.json', df, labels)
    """)
