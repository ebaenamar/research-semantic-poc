"""
Domain-Aware Clustering
Hierarchical clustering that respects medical domain boundaries
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict


class DomainAwareClusterer:
    """
    Two-stage clustering:
    1. Pre-cluster by medical domain (keywords)
    2. Semantic clustering within each domain
    """
    
    def __init__(self):
        # Define medical domains with exclusive keywords
        self.domains = {
            'cardiac': {
                'keywords': ['cardiac', 'heart', 'cardiovascular', 'myocardial', 
                           'coronary', 'ventricular', 'atrial', 'cardiology'],
                'exclude': []  # Will be populated
            },
            'neurological': {
                'keywords': ['brain', 'neural', 'neurological', 'cerebral', 
                           'cognitive', 'neurology', 'stroke', 'seizure'],
                'exclude': []
            },
            'respiratory': {
                'keywords': ['lung', 'pulmonary', 'respiratory', 'airway', 
                           'breathing', 'asthma', 'pneumonia'],
                'exclude': []
            },
            'gastrointestinal': {
                'keywords': ['gastro', 'intestinal', 'digestive', 'liver', 
                           'hepatic', 'bowel', 'stomach'],
                'exclude': []
            },
            'renal': {
                'keywords': ['kidney', 'renal', 'nephro', 'urinary', 'dialysis'],
                'exclude': []
            },
            'hematological': {
                'keywords': ['blood', 'hematological', 'anemia', 'leukemia', 
                           'coagulation', 'platelet'],
                'exclude': []
            },
            'oncological': {
                'keywords': ['cancer', 'tumor', 'oncology', 'malignancy', 
                           'chemotherapy', 'radiation'],
                'exclude': []
            },
            'infectious': {
                'keywords': ['infection', 'infectious', 'sepsis', 'bacterial', 
                           'viral', 'antibiotic'],
                'exclude': []
            },
            'metabolic': {
                'keywords': ['diabetes', 'metabolic', 'endocrine', 'thyroid', 
                           'glucose', 'insulin'],
                'exclude': []
            },
            'immunological': {
                'keywords': ['immune', 'immunology', 'autoimmune', 'allergy', 
                           'inflammation'],
                'exclude': []
            },
            'developmental': {
                'keywords': ['development', 'developmental', 'growth', 'congenital', 
                           'prenatal', 'neonatal'],
                'exclude': []
            },
            'genetic': {
                'keywords': ['genetic', 'genomic', 'mutation', 'hereditary', 
                           'chromosome', 'gene'],
                'exclude': []
            }
        }
        
        # Add exclusions (keywords from other domains)
        for domain_name, domain_info in self.domains.items():
            exclude_keywords = []
            for other_domain, other_info in self.domains.items():
                if other_domain != domain_name:
                    exclude_keywords.extend(other_info['keywords'])
            domain_info['exclude'] = exclude_keywords
    
    def assign_domains(
        self, 
        df: pd.DataFrame, 
        text_columns: List[str] = ['title', 'abstract_text']
    ) -> pd.Series:
        """
        Assign each paper to a medical domain based on keywords
        
        Returns:
            Series with domain labels ('cardiac', 'neurological', 'multi_domain', 'unclassified')
        """
        
        domain_assignments = []
        
        for idx, row in df.iterrows():
            # Combine text from specified columns
            text = ''
            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    text += ' ' + str(row[col])
            
            text_lower = text.lower()
            
            # Score each domain
            domain_scores = {}
            for domain_name, domain_info in self.domains.items():
                # Count matching keywords
                matches = sum(1 for kw in domain_info['keywords'] if kw in text_lower)
                
                # Penalize if other domain keywords present
                conflicts = sum(1 for kw in domain_info['exclude'] if kw in text_lower)
                
                # Net score
                domain_scores[domain_name] = matches - (conflicts * 0.5)
            
            # Assign domain
            max_score = max(domain_scores.values())
            
            if max_score <= 0:
                domain_assignments.append('unclassified')
            elif max_score < 2:
                # Weak signal - might be multi-domain
                top_domains = [d for d, s in domain_scores.items() if s == max_score]
                if len(top_domains) > 1:
                    domain_assignments.append('multi_domain')
                else:
                    domain_assignments.append(top_domains[0])
            else:
                # Strong signal
                top_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
                domain_assignments.append(top_domain)
        
        return pd.Series(domain_assignments, index=df.index)
    
    def cluster_within_domains(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        domain_labels: pd.Series,
        clusterer,  # SemanticClusterer instance
        min_cluster_size: int = 10
    ) -> np.ndarray:
        """
        Perform clustering within each domain separately
        
        Returns:
            Global cluster labels (domain-aware)
        """
        
        global_labels = np.full(len(df), -1, dtype=int)
        next_cluster_id = 0
        
        # Get unique domains (excluding unclassified for now)
        domains_to_cluster = [d for d in domain_labels.unique() 
                             if d not in ['unclassified', 'multi_domain']]
        
        print(f"\n{'='*70}")
        print("DOMAIN-AWARE CLUSTERING")
        print(f"{'='*70}")
        
        for domain in sorted(domains_to_cluster):
            domain_mask = domain_labels == domain
            domain_size = domain_mask.sum()
            
            if domain_size < min_cluster_size:
                print(f"\n⚠️  {domain.upper()}: {domain_size} papers (too small, skipping)")
                continue
            
            print(f"\n🔬 {domain.upper()}: {domain_size} papers")
            
            # Get embeddings for this domain
            domain_embeddings = embeddings[domain_mask]
            
            # Cluster within domain
            try:
                # Reduce dimensions
                reduced = clusterer.reduce_dimensions(
                    domain_embeddings,
                    n_components=min(10, domain_size // 2)
                )
                
                # Cluster
                local_labels = clusterer.cluster_hdbscan(
                    reduced,
                    min_cluster_size=max(5, min_cluster_size // 2),
                    min_samples=3
                )
                
                # Map local labels to global labels
                unique_local = np.unique(local_labels)
                n_clusters = len([l for l in unique_local if l != -1])
                n_noise = (local_labels == -1).sum()
                
                print(f"   ✅ Found {n_clusters} clusters, {n_noise} noise points")
                
                # Assign global labels
                for local_label in unique_local:
                    if local_label == -1:
                        continue  # Keep as noise
                    
                    local_mask = local_labels == local_label
                    global_mask = np.where(domain_mask)[0][local_mask]
                    global_labels[global_mask] = next_cluster_id
                    next_cluster_id += 1
                
            except Exception as e:
                print(f"   ❌ Error clustering {domain}: {str(e)}")
        
        # Handle multi-domain and unclassified
        multi_domain_mask = domain_labels.isin(['multi_domain', 'unclassified'])
        if multi_domain_mask.sum() > 0:
            print(f"\n📊 MULTI-DOMAIN/UNCLASSIFIED: {multi_domain_mask.sum()} papers")
            print("   (Kept as noise for manual review)")
        
        print(f"\n{'='*70}")
        print(f"TOTAL CLUSTERS: {next_cluster_id}")
        print(f"TOTAL NOISE: {(global_labels == -1).sum()}")
        print(f"{'='*70}\n")
        
        return global_labels
    
    def get_domain_statistics(
        self,
        df: pd.DataFrame,
        domain_labels: pd.Series
    ) -> Dict:
        """Get statistics about domain distribution"""
        
        stats = {
            'domain_counts': domain_labels.value_counts().to_dict(),
            'total_papers': len(df),
            'classified_papers': (domain_labels != 'unclassified').sum(),
            'multi_domain_papers': (domain_labels == 'multi_domain').sum()
        }
        
        return stats
    
    def validate_domain_purity(
        self,
        df: pd.DataFrame,
        cluster_labels: np.ndarray,
        domain_labels: pd.Series
    ) -> Dict:
        """
        Validate that clusters don't mix incompatible domains
        
        Returns:
            Dict with purity scores and violations
        """
        
        violations = []
        cluster_purities = {}
        
        unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_domains = domain_labels[cluster_mask]
            
            # Count domains in cluster
            domain_counts = cluster_domains.value_counts()
            
            # Purity = proportion of most common domain
            purity = domain_counts.iloc[0] / len(cluster_domains) if len(domain_counts) > 0 else 0
            cluster_purities[cluster_id] = purity
            
            # Check for violations (multiple strong domains)
            if len(domain_counts) > 1:
                # If second domain has >20% representation, it's a violation
                if domain_counts.iloc[1] / len(cluster_domains) > 0.2:
                    violations.append({
                        'cluster_id': cluster_id,
                        'size': len(cluster_domains),
                        'domains': domain_counts.to_dict(),
                        'purity': purity
                    })
        
        return {
            'cluster_purities': cluster_purities,
            'mean_purity': np.mean(list(cluster_purities.values())) if cluster_purities else 0,
            'violations': violations,
            'n_violations': len(violations)
        }
