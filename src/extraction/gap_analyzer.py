"""
Gap Analyzer Module
Identifies research gaps, contradictions, and opportunities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from collections import Counter
import json
import re


class GapAnalyzer:
    """
    Analyze research landscape to identify gaps and opportunities
    """
    
    def __init__(self):
        self.gaps = {}
        self.contradictions = []
        self.opportunities = []
        
    def analyze_temporal_gaps(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        year_column: str = 'year'
    ) -> Dict:
        """
        Identify temporal gaps (outdated research areas)
        
        Args:
            df: DataFrame with paper data
            labels: Cluster labels
            year_column: Column name for publication year
            
        Returns:
            Dictionary with temporal analysis
        """
        print("Analyzing temporal gaps...")
        
        temporal_gaps = {}
        current_year = pd.Timestamp.now().year
        
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
                
            mask = labels == cluster_id
            cluster_df = df[mask]
            
            if year_column in cluster_df.columns:
                years = pd.to_numeric(cluster_df[year_column], errors='coerce')
                valid_years = years.dropna()
                
                if len(valid_years) > 0:
                    avg_year = valid_years.mean()
                    years_since_peak = current_year - valid_years.max()
                    
                    temporal_gaps[f"cluster_{cluster_id}"] = {
                        'average_year': float(avg_year),
                        'most_recent': int(valid_years.max()),
                        'oldest': int(valid_years.min()),
                        'years_since_peak': int(years_since_peak),
                        'is_outdated': years_since_peak > 5,
                        'needs_update': years_since_peak > 3
                    }
        
        return temporal_gaps
    
    def analyze_methodological_gaps(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        text_column: str = 'abstract'
    ) -> Dict:
        """
        Identify methodological gaps by analyzing methodology mentions
        
        Args:
            df: DataFrame with paper data
            labels: Cluster labels
            text_column: Column to analyze for methods
            
        Returns:
            Dictionary with methodological analysis
        """
        print("Analyzing methodological gaps...")
        
        # Common research methodologies to look for
        methodologies = {
            'experimental': ['experiment', 'trial', 'rct', 'randomized'],
            'observational': ['cohort', 'case-control', 'observational', 'longitudinal'],
            'meta_analysis': ['meta-analysis', 'systematic review', 'meta analysis'],
            'computational': ['simulation', 'computational', 'modeling', 'algorithm'],
            'qualitative': ['qualitative', 'interview', 'survey', 'questionnaire'],
            'machine_learning': ['machine learning', 'deep learning', 'neural network', 'ai'],
            'genomics': ['genomic', 'sequencing', 'genome-wide', 'gwas'],
            'imaging': ['mri', 'ct scan', 'imaging', 'ultrasound']
        }
        
        method_gaps = {}
        
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
                
            mask = labels == cluster_id
            cluster_df = df[mask]
            
            if text_column not in cluster_df.columns:
                continue
            
            # Count methodology mentions
            method_counts = {method: 0 for method in methodologies}
            
            for text in cluster_df[text_column].dropna():
                text_lower = text.lower()
                for method_type, keywords in methodologies.items():
                    if any(keyword in text_lower for keyword in keywords):
                        method_counts[method_type] += 1
            
            # Identify underused methods
            total_papers = len(cluster_df)
            method_percentages = {
                method: (count / total_papers * 100) 
                for method, count in method_counts.items()
            }
            
            underused = [
                method for method, pct in method_percentages.items() 
                if pct < 10
            ]
            
            method_gaps[f"cluster_{cluster_id}"] = {
                'size': total_papers,
                'method_usage': method_percentages,
                'underused_methods': underused,
                'dominant_methods': sorted(
                    method_percentages.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
            }
        
        return method_gaps
    
    def detect_contradictions(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        embeddings: np.ndarray
    ) -> List[Dict]:
        """
        Detect potential contradictions between papers
        Uses semantic similarity and keyword analysis
        
        Args:
            df: DataFrame with paper data
            labels: Cluster labels
            embeddings: Paper embeddings
            
        Returns:
            List of potential contradictions
        """
        print("Detecting contradictions...")
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        contradictions = []
        
        # Words indicating contradictory findings
        contradiction_keywords = [
            'contradict', 'however', 'contrary', 'opposite',
            'conflict', 'disagree', 'refute', 'challenge'
        ]
        
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            
            mask = labels == cluster_id
            cluster_indices = np.where(mask)[0]
            
            if len(cluster_indices) < 2:
                continue
            
            cluster_embeddings = embeddings[cluster_indices]
            
            # Find papers with high similarity but potentially contradictory content
            similarities = cosine_similarity(cluster_embeddings)
            
            for i in range(len(cluster_indices)):
                for j in range(i+1, len(cluster_indices)):
                    sim_score = similarities[i, j]
                    
                    # High similarity suggests same topic
                    if sim_score > 0.7:
                        idx_i = cluster_indices[i]
                        idx_j = cluster_indices[j]
                        
                        abstract_i = str(df.iloc[idx_i].get('abstract', '')).lower()
                        abstract_j = str(df.iloc[idx_j].get('abstract', '')).lower()
                        
                        # Check for contradiction keywords
                        has_contradiction = any(
                            keyword in abstract_i or keyword in abstract_j
                            for keyword in contradiction_keywords
                        )
                        
                        if has_contradiction:
                            contradictions.append({
                                'cluster_id': cluster_id,
                                'paper_1_idx': int(idx_i),
                                'paper_2_idx': int(idx_j),
                                'paper_1_title': df.iloc[idx_i].get('title', ''),
                                'paper_2_title': df.iloc[idx_j].get('title', ''),
                                'similarity_score': float(sim_score),
                                'requires_investigation': True
                            })
        
        return contradictions
    
    def identify_cross_cluster_opportunities(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        embeddings: np.ndarray,
        min_similarity: float = 0.5
    ) -> List[Dict]:
        """
        Identify opportunities by finding connections between different clusters
        
        Args:
            df: DataFrame with paper data
            labels: Cluster labels
            embeddings: Paper embeddings
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of cross-cluster opportunities
        """
        print("Identifying cross-cluster opportunities...")
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        opportunities = []
        
        unique_clusters = [c for c in set(labels) if c != -1]
        
        for i, cluster_i in enumerate(unique_clusters):
            for cluster_j in unique_clusters[i+1:]:
                mask_i = labels == cluster_i
                mask_j = labels == cluster_j
                
                embeddings_i = embeddings[mask_i]
                embeddings_j = embeddings[mask_j]
                
                # Compute cross-cluster similarities
                cross_sim = cosine_similarity(embeddings_i, embeddings_j)
                
                # Find papers with moderate similarity (potential for novel combination)
                similar_pairs = np.argwhere(
                    (cross_sim > min_similarity) & (cross_sim < 0.8)
                )
                
                if len(similar_pairs) > 5:  # Significant overlap
                    indices_i = np.where(mask_i)[0]
                    indices_j = np.where(mask_j)[0]
                    
                    opportunities.append({
                        'cluster_1': int(cluster_i),
                        'cluster_2': int(cluster_j),
                        'n_connections': len(similar_pairs),
                        'opportunity_type': 'methodological_transfer',
                        'description': (
                            f"Methods from cluster {cluster_i} could be "
                            f"applied to problems in cluster {cluster_j}"
                        ),
                        'sample_papers': {
                            'cluster_1': df.iloc[indices_i[0]].get('title', ''),
                            'cluster_2': df.iloc[indices_j[0]].get('title', '')
                        }
                    })
        
        return opportunities
    
    def generate_gap_report(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        embeddings: np.ndarray
    ) -> Dict:
        """
        Generate comprehensive gap analysis report
        
        Args:
            df: DataFrame with paper data
            labels: Cluster labels
            embeddings: Paper embeddings
            
        Returns:
            Complete gap analysis
        """
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE GAP ANALYSIS")
        print("="*60 + "\n")
        
        report = {
            'summary': {
                'total_papers': len(df),
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'n_noise': int((labels == -1).sum())
            },
            'temporal_gaps': self.analyze_temporal_gaps(df, labels),
            'methodological_gaps': self.analyze_methodological_gaps(df, labels),
            'contradictions': self.detect_contradictions(df, labels, embeddings),
            'cross_cluster_opportunities': self.identify_cross_cluster_opportunities(
                df, labels, embeddings
            )
        }
        
        # Generate actionable insights
        report['actionable_insights'] = self._generate_insights(report)
        
        return report
    
    def _generate_insights(self, report: Dict) -> List[str]:
        """Generate actionable insights from gap analysis"""
        insights = []
        
        # Temporal insights
        outdated_clusters = [
            k for k, v in report['temporal_gaps'].items()
            if v.get('is_outdated', False)
        ]
        if outdated_clusters:
            insights.append(
                f"⚠️  {len(outdated_clusters)} clusters have outdated research "
                f"(>5 years old) - opportunity for updated studies"
            )
        
        # Methodological insights
        if report['methodological_gaps']:
            insights.append(
                "🔬 Multiple clusters show underutilization of modern methods "
                "(ML, computational) - opportunity for methodological innovation"
            )
        
        # Contradiction insights
        if report['contradictions']:
            insights.append(
                f"⚡ Found {len(report['contradictions'])} potential contradictions "
                f"requiring investigation - opportunity for replication studies"
            )
        
        # Cross-cluster insights
        if report['cross_cluster_opportunities']:
            insights.append(
                f"🔗 Identified {len(report['cross_cluster_opportunities'])} "
                f"cross-cluster opportunities - potential for novel combinations"
            )
        
        return insights
    
    def save_report(self, report: Dict, filepath: str):
        """Save gap analysis report to JSON"""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Gap analysis report saved to {filepath}")
        print("\nKey Insights:")
        for insight in report.get('actionable_insights', []):
            print(f"  {insight}")


if __name__ == "__main__":
    print("Gap Analyzer Module")
    print("=" * 50)
    print("\nIdentifies:")
    print("  - Temporal gaps (outdated research)")
    print("  - Methodological gaps (underused methods)")
    print("  - Contradictions (conflicting findings)")
    print("  - Cross-cluster opportunities (novel combinations)")
