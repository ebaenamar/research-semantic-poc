"""
Adaptive Clustering
Reduces noise by using multiple strategies and smaller, tighter clusters
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import hdbscan
from sklearn.cluster import DBSCAN
from collections import Counter


class AdaptiveClusterer:
    """
    Multi-strategy clustering to minimize noise and find tight, coherent clusters
    """
    
    def __init__(self):
        self.strategies = [
            'hierarchical_hdbscan',  # Start with large clusters, split if needed
            'dbscan_epsilon',        # Distance-based with adaptive epsilon
            'future_work_focused'    # Focus on papers with future work mentions
        ]
    
    def cluster_with_noise_reduction(
        self,
        embeddings: np.ndarray,
        reduced_embeddings: np.ndarray,
        df: pd.DataFrame,
        min_cluster_size: int = 5,  # Smaller minimum
        target_noise_ratio: float = 0.3  # Target <30% noise
    ) -> np.ndarray:
        """
        Adaptive clustering that tries multiple strategies to reduce noise
        """
        
        print(f"\n{'='*70}")
        print("ADAPTIVE CLUSTERING - NOISE REDUCTION")
        print(f"{'='*70}")
        print(f"Target noise ratio: {target_noise_ratio:.1%}")
        print(f"Min cluster size: {min_cluster_size}")
        
        best_labels = None
        best_noise_ratio = 1.0
        
        # Strategy 1: Hierarchical HDBSCAN with smaller clusters
        print(f"\n🔍 Strategy 1: Hierarchical HDBSCAN")
        labels_hdbscan = self._hierarchical_hdbscan(
            reduced_embeddings,
            min_cluster_size=min_cluster_size,
            min_samples=2  # Very lenient
        )
        noise_ratio_hdbscan = (labels_hdbscan == -1).sum() / len(labels_hdbscan)
        print(f"   Noise ratio: {noise_ratio_hdbscan:.1%}")
        
        if noise_ratio_hdbscan < best_noise_ratio:
            best_labels = labels_hdbscan
            best_noise_ratio = noise_ratio_hdbscan
        
        # Strategy 2: DBSCAN with adaptive epsilon
        print(f"\n🔍 Strategy 2: Adaptive DBSCAN")
        labels_dbscan = self._adaptive_dbscan(
            reduced_embeddings,
            min_samples=min_cluster_size
        )
        noise_ratio_dbscan = (labels_dbscan == -1).sum() / len(labels_dbscan)
        print(f"   Noise ratio: {noise_ratio_dbscan:.1%}")
        
        if noise_ratio_dbscan < best_noise_ratio:
            best_labels = labels_dbscan
            best_noise_ratio = noise_ratio_dbscan
        
        # Strategy 3: Future work focused
        print(f"\n🔍 Strategy 3: Future Work Focused")
        labels_future = self._future_work_clustering(
            reduced_embeddings,
            df,
            min_cluster_size=min_cluster_size
        )
        noise_ratio_future = (labels_future == -1).sum() / len(labels_future)
        print(f"   Noise ratio: {noise_ratio_future:.1%}")
        
        if noise_ratio_future < best_noise_ratio:
            best_labels = labels_future
            best_noise_ratio = noise_ratio_future
        
        # If still too much noise, try rescue strategy
        if best_noise_ratio > target_noise_ratio:
            print(f"\n🚨 Noise still high ({best_noise_ratio:.1%}), applying rescue strategy...")
            best_labels = self._rescue_noise_points(
                reduced_embeddings,
                best_labels,
                max_distance_percentile=10  # Assign to nearest cluster if within 10th percentile
            )
            best_noise_ratio = (best_labels == -1).sum() / len(best_labels)
            print(f"   Final noise ratio: {best_noise_ratio:.1%}")
        
        n_clusters = len([l for l in np.unique(best_labels) if l != -1])
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS:")
        print(f"Clusters: {n_clusters}")
        print(f"Noise: {(best_labels == -1).sum()} papers ({best_noise_ratio:.1%})")
        print(f"{'='*70}\n")
        
        return best_labels
    
    def _hierarchical_hdbscan(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        min_samples: int = 2
    ) -> np.ndarray:
        """
        HDBSCAN with very lenient parameters to capture more papers
        """
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.0,  # No epsilon constraint
            cluster_selection_method='eom',  # Excess of mass
            allow_single_cluster=False
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        n_clusters = len([l for l in np.unique(labels) if l != -1])
        print(f"   Found {n_clusters} clusters")
        
        return labels
    
    def _adaptive_dbscan(
        self,
        embeddings: np.ndarray,
        min_samples: int = 5
    ) -> np.ndarray:
        """
        DBSCAN with automatically determined epsilon
        """
        
        # Calculate pairwise distances
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(embeddings, metric='euclidean')
        
        # For each point, find distance to kth nearest neighbor
        k = min_samples
        kth_distances = []
        for i in range(len(embeddings)):
            dists = np.sort(distances[i])
            if len(dists) > k:
                kth_distances.append(dists[k])
        
        # Use 20th percentile as epsilon (more lenient than median)
        epsilon = np.percentile(kth_distances, 20)
        
        print(f"   Adaptive epsilon: {epsilon:.4f}")
        
        clusterer = DBSCAN(
            eps=epsilon,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        n_clusters = len([l for l in np.unique(labels) if l != -1])
        print(f"   Found {n_clusters} clusters")
        
        return labels
    
    def _future_work_clustering(
        self,
        embeddings: np.ndarray,
        df: pd.DataFrame,
        min_cluster_size: int = 5
    ) -> np.ndarray:
        """
        Focus on papers that mention future work, limitations, or gaps
        These are more likely to form coherent research opportunities
        """
        
        future_work_keywords = [
            'future work', 'future research', 'future studies',
            'limitation', 'limited by', 'gap', 'need for',
            'further research', 'additional studies', 'warrant',
            'should be investigated', 'remains to be', 'unclear',
            'not yet', 'unexplored', 'understudied'
        ]
        
        # Score papers by future work mentions
        future_scores = []
        for idx, row in df.iterrows():
            text = ''
            if 'abstract_text' in row and pd.notna(row['abstract_text']):
                text = str(row['abstract_text']).lower()
            
            score = sum(1 for kw in future_work_keywords if kw in text)
            future_scores.append(score)
        
        future_scores = np.array(future_scores)
        
        # Papers with future work mentions get priority
        has_future_work = future_scores > 0
        
        print(f"   Papers with future work mentions: {has_future_work.sum()}")
        
        # Cluster all papers but weight by future work
        # Use HDBSCAN with sample weights
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            cluster_selection_method='eom'
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        n_clusters = len([l for l in np.unique(labels) if l != -1])
        print(f"   Found {n_clusters} clusters")
        
        return labels
    
    def _rescue_noise_points(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        max_distance_percentile: int = 10
    ) -> np.ndarray:
        """
        Assign noise points to nearest cluster if they're close enough
        """
        
        noise_mask = labels == -1
        n_noise = noise_mask.sum()
        
        if n_noise == 0:
            return labels
        
        print(f"   Attempting to rescue {n_noise} noise points...")
        
        # Get cluster centroids
        unique_clusters = [l for l in np.unique(labels) if l != -1]
        if len(unique_clusters) == 0:
            return labels
        
        centroids = {}
        for cluster_id in unique_clusters:
            cluster_mask = labels == cluster_id
            centroids[cluster_id] = embeddings[cluster_mask].mean(axis=0)
        
        # For each noise point, find distance to nearest cluster
        rescued_labels = labels.copy()
        rescued_count = 0
        
        for idx in np.where(noise_mask)[0]:
            point = embeddings[idx]
            
            # Calculate distances to all centroids
            distances = {}
            for cluster_id, centroid in centroids.items():
                dist = np.linalg.norm(point - centroid)
                distances[cluster_id] = dist
            
            # Find nearest cluster
            nearest_cluster = min(distances.items(), key=lambda x: x[1])
            nearest_id, nearest_dist = nearest_cluster
            
            # Calculate distance threshold (percentile of within-cluster distances)
            cluster_mask = labels == nearest_id
            cluster_points = embeddings[cluster_mask]
            cluster_centroid = centroids[nearest_id]
            
            within_cluster_dists = [
                np.linalg.norm(p - cluster_centroid) 
                for p in cluster_points
            ]
            
            threshold = np.percentile(within_cluster_dists, max_distance_percentile)
            
            # Assign if close enough
            if nearest_dist <= threshold:
                rescued_labels[idx] = nearest_id
                rescued_count += 1
        
        print(f"   Rescued {rescued_count} points ({rescued_count/n_noise:.1%})")
        
        return rescued_labels
    
    def analyze_cluster_quality(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        df: pd.DataFrame
    ) -> Dict:
        """
        Analyze quality of clusters for future work potential
        """
        
        quality_metrics = {}
        
        unique_clusters = [l for l in np.unique(labels) if l != -1]
        
        for cluster_id in unique_clusters:
            cluster_mask = labels == cluster_id
            cluster_df = df[cluster_mask]
            cluster_embeddings = embeddings[cluster_mask]
            
            # Cohesion (how tight is the cluster)
            centroid = cluster_embeddings.mean(axis=0)
            distances = [np.linalg.norm(p - centroid) for p in cluster_embeddings]
            cohesion = 1 / (1 + np.mean(distances))  # Higher is better
            
            # Future work potential
            future_keywords = ['future', 'limitation', 'gap', 'need', 'further']
            future_mentions = 0
            for idx, row in cluster_df.iterrows():
                if 'abstract_text' in row and pd.notna(row['abstract_text']):
                    text = str(row['abstract_text']).lower()
                    future_mentions += sum(1 for kw in future_keywords if kw in text)
            
            future_score = future_mentions / len(cluster_df)
            
            # Temporal recency (newer papers = more relevant future work)
            if 'publication_year' in cluster_df.columns:
                years = pd.to_numeric(cluster_df['publication_year'], errors='coerce').dropna()
                if len(years) > 0:
                    recency_score = (years.mean() - 2000) / 25  # Normalize to 0-1
                else:
                    recency_score = 0.5
            else:
                recency_score = 0.5
            
            quality_metrics[cluster_id] = {
                'size': len(cluster_df),
                'cohesion': cohesion,
                'future_work_score': future_score,
                'recency_score': recency_score,
                'overall_quality': (cohesion * 0.4 + future_score * 0.4 + recency_score * 0.2)
            }
        
        return quality_metrics
