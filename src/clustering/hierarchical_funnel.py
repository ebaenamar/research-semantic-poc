"""
Hierarchical Funnel Clustering
Progressive refinement: Topic → Methodology → Temporal → Semantic
Each stage applies conditional probability filtering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import hdbscan
from sklearn.cluster import DBSCAN
import re


class HierarchicalFunnelClusterer:
    """
    Multi-stage funnel clustering with importance weighting
    
    Stage 1 (Most Important): Specific Topic/Domain
    Stage 2: Methodology Coherence  
    Stage 3: Temporal Coherence (Recency)
    Stage 4: Fine-grained Semantic Clustering
    
    Each stage filters based on conditional probability
    """
    
    def __init__(self):
        # Define importance weights (sum to 1.0)
        self.importance_weights = {
            'topic': 0.40,        # Most important: must be same specific topic
            'methodology': 0.25,  # Second: same research method
            'temporal': 0.15,     # Third: similar time period
            'semantic': 0.20      # Fourth: semantic similarity
        }
        
        # Specific medical topics (from thematic_coherence.py)
        self.specific_topics = {
            'heart_failure': ['heart failure', 'cardiac failure', 'congestive heart', 'hf'],
            'arrhythmia': ['arrhythmia', 'atrial fibrillation', 'afib', 'ventricular tachycardia', 'vtach'],
            'stroke': ['stroke', 'cerebrovascular', 'ischemic brain', 'cva'],
            'diabetes': ['diabetes', 'diabetic', 'glycemic control', 't2dm', 't1dm'],
            'kidney_disease': ['chronic kidney disease', 'ckd', 'renal insufficiency', 'esrd'],
            'aki': ['acute kidney injury', 'aki', 'acute renal failure'],
            'sepsis': ['sepsis', 'septic shock', 'severe infection'],
            'cancer_lung': ['lung cancer', 'pulmonary carcinoma', 'nsclc', 'sclc'],
            'cancer_breast': ['breast cancer', 'mammary carcinoma', 'breast neoplasm'],
            'covid': ['covid', 'sars-cov-2', 'coronavirus disease', 'covid-19'],
            'alzheimer': ['alzheimer', 'dementia', 'cognitive decline', 'ad'],
            'parkinsons': ['parkinson', 'pd', 'parkinsonian'],
            'epilepsy': ['epilepsy', 'seizure', 'epileptic'],
            'asthma': ['asthma', 'bronchial asthma'],
            'copd': ['copd', 'chronic obstructive pulmonary'],
            'hypertension': ['hypertension', 'high blood pressure', 'htn'],
            'mi': ['myocardial infarction', 'heart attack', 'mi', 'stemi', 'nstemi'],
            'pneumonia': ['pneumonia', 'pulmonary infection'],
            'obesity': ['obesity', 'obese', 'bmi', 'overweight'],
            'depression': ['depression', 'depressive disorder', 'mdd'],
            'ecg': ['ecg', 'electrocardiogram', 'ekg', 'cardiac rhythm'],
            'mri': ['mri', 'magnetic resonance', 'brain imaging'],
            'ct_scan': ['ct scan', 'computed tomography', 'ct imaging'],
            'biomarker': ['biomarker', 'biological marker', 'serum marker'],
            'mortality': ['mortality', 'death', 'survival', 'fatal'],
            'readmission': ['readmission', 're-admission', 'rehospitalization'],
            'icu': ['icu', 'intensive care', 'critical care'],
            'pediatric': ['pediatric', 'children', 'child', 'infant', 'neonatal'],
            'geriatric': ['geriatric', 'elderly', 'older adult', 'aging'],
        }
        
        # Methodology keywords
        self.methodology_types = {
            'rct': ['randomized controlled trial', 'rct', 'randomized trial'],
            'cohort': ['cohort study', 'prospective cohort', 'retrospective cohort'],
            'case_control': ['case-control', 'case control study'],
            'meta_analysis': ['meta-analysis', 'systematic review', 'meta analysis'],
            'machine_learning': ['machine learning', 'deep learning', 'neural network', 'ml model'],
            'observational': ['observational study', 'observational'],
            'cross_sectional': ['cross-sectional', 'cross sectional'],
            'clinical_trial': ['clinical trial', 'phase i', 'phase ii', 'phase iii'],
            'registry': ['registry', 'registry-based', 'database study'],
            'imaging': ['imaging study', 'radiological', 'ct', 'mri', 'ultrasound'],
            'genomic': ['genomic', 'genetic', 'genome-wide', 'gwas'],
            'biomarker': ['biomarker study', 'biomarker analysis'],
        }
    
    def cluster_hierarchical_funnel(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        reduced_embeddings: np.ndarray,
        min_cluster_size: int = 5,
        min_topic_coverage: float = 0.4,  # Más permisivo: 40% en vez de 60%
        min_methodology_coverage: float = 0.3,  # Más permisivo: 30% en vez de 50%
        recency_window_years: int = 10  # Ventana más amplia: 10 años
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply hierarchical funnel clustering
        
        Returns:
            labels: Cluster assignments
            report: Detailed funnel report
        """
        
        n_papers = len(df)
        labels = np.full(n_papers, -1, dtype=int)
        
        report = {
            'stages': {},
            'funnel_summary': {},
            'cluster_details': {}
        }
        
        # ========================================
        # STAGE 1: TOPIC FILTERING (40% importance)
        # ========================================
        print("\n🔍 Stage 1: Topic Assignment (40% importance)")
        
        topic_assignments = self._assign_specific_topics(df)
        
        # Group by topic
        topic_groups = defaultdict(list)
        for idx, topic in enumerate(topic_assignments):
            if topic != 'unclassified':
                topic_groups[topic].append(idx)
        
        report['stages']['stage1_topic'] = {
            'n_topics_found': len(topic_groups),
            'topic_sizes': {t: len(indices) for t, indices in topic_groups.items()},
            'unclassified': (topic_assignments == 'unclassified').sum()
        }
        
        print(f"   Found {len(topic_groups)} specific topics")
        print(f"   Unclassified: {(topic_assignments == 'unclassified').sum()}")
        
        # ========================================
        # STAGE 2: METHODOLOGY FILTERING (25% importance)
        # ========================================
        print("\n📊 Stage 2: Methodology Coherence (25% importance)")
        
        methodology_assignments = self._assign_methodologies(df)
        
        # Within each topic, group by methodology
        topic_method_groups = {}
        for topic, topic_indices in topic_groups.items():
            if len(topic_indices) < min_cluster_size:
                continue
            
            # Group by methodology within this topic
            method_groups = defaultdict(list)
            for idx in topic_indices:
                method = methodology_assignments[idx]
                if method != 'unclassified':
                    method_groups[method].append(idx)
            
            topic_method_groups[topic] = method_groups
        
        report['stages']['stage2_methodology'] = {
            'topic_method_groups': {
                topic: {m: len(indices) for m, indices in methods.items()}
                for topic, methods in topic_method_groups.items()
            }
        }
        
        print(f"   Created {sum(len(m) for m in topic_method_groups.values())} topic+method groups")
        
        # ========================================
        # STAGE 3: TEMPORAL FILTERING (15% importance)
        # ========================================
        print("\n📅 Stage 3: Temporal Coherence (15% importance)")
        
        # Within each topic+method group, filter by recency
        topic_method_time_groups = {}
        for topic, method_groups in topic_method_groups.items():
            topic_method_time_groups[topic] = {}
            
            for method, method_indices in method_groups.items():
                if len(method_indices) < min_cluster_size:
                    continue
                
                # Get years
                years = []
                valid_indices = []
                for idx in method_indices:
                    if 'publication_year' in df.columns:
                        year = pd.to_numeric(df.iloc[idx]['publication_year'], errors='coerce')
                        if pd.notna(year):
                            years.append(year)
                            valid_indices.append(idx)
                
                if not years:
                    continue
                
                # Group by time windows
                max_year = max(years)
                time_groups = defaultdict(list)
                
                for idx, year in zip(valid_indices, years):
                    # Recent papers (within window)
                    if max_year - year <= recency_window_years:
                        time_groups['recent'].append(idx)
                    else:
                        time_groups['older'].append(idx)
                
                topic_method_time_groups[topic][method] = time_groups
        
        report['stages']['stage3_temporal'] = {
            'recency_window': recency_window_years,
            'time_groups_created': sum(
                len(time_groups) 
                for methods in topic_method_time_groups.values() 
                for time_groups in methods.values()
            )
        }
        
        print(f"   Applied {recency_window_years}-year recency window")
        
        # ========================================
        # STAGE 4: SEMANTIC CLUSTERING (20% importance)
        # ========================================
        print("\n🧬 Stage 4: Fine-grained Semantic Clustering (20% importance)")
        
        cluster_id = 0
        cluster_details = {}
        
        # For each topic+method+time group, do semantic clustering
        for topic, method_groups in topic_method_time_groups.items():
            for method, time_groups in method_groups.items():
                for time_label, indices in time_groups.items():
                    if len(indices) < min_cluster_size:
                        continue
                    
                    # Get embeddings for this group
                    group_embeddings = reduced_embeddings[indices]
                    
                    # Semantic clustering with HDBSCAN
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=2,
                        metric='euclidean',
                        cluster_selection_method='eom'
                    )
                    
                    local_labels = clusterer.fit_predict(group_embeddings)
                    
                    # Map to global labels
                    for local_label in np.unique(local_labels):
                        if local_label == -1:
                            continue
                        
                        local_mask = local_labels == local_label
                        global_indices = np.array(indices)[local_mask]
                        
                        if len(global_indices) >= min_cluster_size:
                            labels[global_indices] = cluster_id
                            
                            # Store cluster details
                            cluster_details[cluster_id] = {
                                'topic': topic,
                                'methodology': method,
                                'time_group': time_label,
                                'size': len(global_indices),
                                'indices': global_indices.tolist()
                            }
                            
                            cluster_id += 1
        
        report['stages']['stage4_semantic'] = {
            'final_clusters': cluster_id,
            'clustered_papers': (labels != -1).sum(),
            'noise': (labels == -1).sum()
        }
        
        report['cluster_details'] = cluster_details
        
        # Calculate funnel summary
        report['funnel_summary'] = self._calculate_funnel_summary(
            df, labels, topic_assignments, methodology_assignments, cluster_details
        )
        
        print(f"\n✅ Funnel complete:")
        print(f"   - Final clusters: {cluster_id}")
        print(f"   - Clustered: {(labels != -1).sum()}/{n_papers} ({(labels != -1).sum()/n_papers*100:.1f}%)")
        print(f"   - Noise: {(labels == -1).sum()} ({(labels == -1).sum()/n_papers*100:.1f}%)")
        
        return labels, report
    
    def _assign_specific_topics(self, df: pd.DataFrame) -> np.ndarray:
        """Stage 1: Assign specific medical topics"""
        
        topics = []
        
        for idx, row in df.iterrows():
            text = ''
            if 'title' in row and pd.notna(row['title']):
                text += str(row['title']).lower() + ' '
            if 'abstract_text' in row and pd.notna(row['abstract_text']):
                text += str(row['abstract_text']).lower()
            
            # Find matching topics
            found_topics = []
            for topic_name, keywords in self.specific_topics.items():
                if any(kw in text for kw in keywords):
                    found_topics.append(topic_name)
            
            # Use most specific topic (prefer specific conditions over general terms)
            if found_topics:
                # Priority: specific conditions > imaging > general
                priority_order = [
                    'heart_failure', 'arrhythmia', 'stroke', 'mi', 'aki', 
                    'sepsis', 'covid', 'alzheimer', 'parkinsons',
                    'diabetes', 'kidney_disease', 'cancer_lung', 'cancer_breast',
                    'ecg', 'mri', 'ct_scan',
                    'pediatric', 'geriatric', 'icu'
                ]
                
                for priority_topic in priority_order:
                    if priority_topic in found_topics:
                        topics.append(priority_topic)
                        break
                else:
                    topics.append(found_topics[0])
            else:
                topics.append('unclassified')
        
        return np.array(topics)
    
    def _assign_methodologies(self, df: pd.DataFrame) -> np.ndarray:
        """Stage 2: Assign research methodologies"""
        
        methods = []
        
        for idx, row in df.iterrows():
            text = ''
            if 'title' in row and pd.notna(row['title']):
                text += str(row['title']).lower() + ' '
            if 'abstract_text' in row and pd.notna(row['abstract_text']):
                text += str(row['abstract_text']).lower()
            
            # Find matching methodologies
            found_methods = []
            for method_name, keywords in self.methodology_types.items():
                if any(kw in text for kw in keywords):
                    found_methods.append(method_name)
            
            # Priority: specific > general
            if found_methods:
                priority_order = [
                    'rct', 'meta_analysis', 'clinical_trial',
                    'machine_learning', 'genomic', 'imaging',
                    'cohort', 'case_control', 'registry',
                    'observational', 'cross_sectional'
                ]
                
                for priority_method in priority_order:
                    if priority_method in found_methods:
                        methods.append(priority_method)
                        break
                else:
                    methods.append(found_methods[0])
            else:
                methods.append('unclassified')
        
        return np.array(methods)
    
    def _calculate_funnel_summary(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        topic_assignments: np.ndarray,
        methodology_assignments: np.ndarray,
        cluster_details: Dict
    ) -> Dict:
        """Calculate summary statistics for the funnel"""
        
        n_total = len(df)
        n_clustered = (labels != -1).sum()
        
        # Topic distribution
        topic_dist = Counter(topic_assignments[topic_assignments != 'unclassified'])
        
        # Methodology distribution
        method_dist = Counter(methodology_assignments[methodology_assignments != 'unclassified'])
        
        # Cluster purity scores
        purity_scores = []
        for cluster_id, details in cluster_details.items():
            # Perfect purity: all papers same topic + same method
            purity = 1.0  # Each cluster is guaranteed pure by construction
            purity_scores.append(purity)
        
        return {
            'total_papers': n_total,
            'clustered_papers': n_clustered,
            'clustering_rate': n_clustered / n_total if n_total > 0 else 0,
            'topic_distribution': dict(topic_dist),
            'methodology_distribution': dict(method_dist),
            'avg_purity': np.mean(purity_scores) if purity_scores else 0,
            'n_clusters': len(cluster_details)
        }
