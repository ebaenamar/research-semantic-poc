"""
Thematic Coherence Validator
Ensures clusters contain papers about the SAME specific topic, not just shared keywords
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter
import re


class ThematicCoherenceValidator:
    """
    Validates that papers in a cluster are truly about the same specific topic
    """
    
    def __init__(self):
        # Specific medical conditions/topics
        self.specific_topics = {
            'heart_failure': ['heart failure', 'cardiac failure', 'congestive heart'],
            'arrhythmia': ['arrhythmia', 'atrial fibrillation', 'ventricular tachycardia'],
            'stroke': ['stroke', 'cerebrovascular', 'ischemic brain'],
            'diabetes': ['diabetes', 'diabetic', 'glycemic control'],
            'kidney_disease': ['chronic kidney disease', 'ckd', 'renal insufficiency'],
            'sepsis': ['sepsis', 'septic shock', 'severe infection'],
            'cancer_lung': ['lung cancer', 'pulmonary carcinoma', 'nsclc'],
            'cancer_breast': ['breast cancer', 'mammary carcinoma'],
            'covid': ['covid', 'sars-cov-2', 'coronavirus disease'],
            'aki': ['acute kidney injury', 'aki', 'acute renal failure'],
            'alzheimer': ['alzheimer', 'dementia', 'cognitive decline'],
            'parkinsons': ['parkinson', 'pd', 'parkinsonian'],
            'epilepsy': ['epilepsy', 'seizure', 'epileptic'],
            'asthma': ['asthma', 'bronchial asthma'],
            'copd': ['copd', 'chronic obstructive pulmonary'],
            'hypertension': ['hypertension', 'high blood pressure'],
            'myocardial_infarction': ['myocardial infarction', 'heart attack', 'mi'],
            'pneumonia': ['pneumonia', 'pulmonary infection'],
            'obesity': ['obesity', 'obese', 'bmi'],
            'depression': ['depression', 'depressive disorder', 'mdd'],
            'ecg': ['ecg', 'electrocardiogram', 'ekg'],
            'mri': ['mri', 'magnetic resonance'],
            'ct_scan': ['ct scan', 'computed tomography'],
            'biomarker': ['biomarker', 'biological marker'],
            'mortality': ['mortality', 'death', 'survival'],
            'readmission': ['readmission', 're-admission', 'rehospitalization'],
            'diagnosis': ['diagnosis', 'diagnostic', 'detection'],
            'prognosis': ['prognosis', 'prognostic', 'outcome prediction'],
            'treatment': ['treatment', 'therapy', 'therapeutic intervention'],
        }
    
    def validate_cluster_coherence(
        self,
        df: pd.DataFrame,
        min_topic_coverage: float = 0.6,
        require_specific_topic: bool = True
    ) -> Dict:
        """
        Validate that papers in cluster share a specific topic
        
        Args:
            df: DataFrame with papers in cluster
            min_topic_coverage: Minimum % of papers that must share the specific topic
            require_specific_topic: If True, cluster must have identifiable specific topic
            
        Returns:
            Dict with validation results
        """
        
        # Combine all text
        all_text = []
        for idx, row in df.iterrows():
            text = ''
            if 'title' in row and pd.notna(row['title']):
                text += str(row['title']) + ' '
            if 'abstract_text' in row and pd.notna(row['abstract_text']):
                text += str(row['abstract_text'])
            all_text.append(text.lower())
        
        # Find specific topics in each paper
        paper_topics = []
        for text in all_text:
            found_topics = []
            for topic_name, keywords in self.specific_topics.items():
                if any(kw in text for kw in keywords):
                    found_topics.append(topic_name)
            paper_topics.append(found_topics)
        
        # Count topic occurrences
        topic_counter = Counter()
        for topics in paper_topics:
            for topic in topics:
                topic_counter[topic] += 1
        
        # Find dominant topic
        if not topic_counter:
            return {
                'coherent': False,
                'reason': 'No specific medical topic identified',
                'dominant_topic': None,
                'coverage': 0.0,
                'requires_split': True
            }
        
        dominant_topic, count = topic_counter.most_common(1)[0]
        coverage = count / len(df)
        
        # Check if coherent
        is_coherent = coverage >= min_topic_coverage
        
        # Check for topic mixing (multiple topics with significant coverage)
        topic_list = topic_counter.most_common()
        mixed_topics = [t for t, c in topic_list if c / len(df) >= 0.3]  # 30% threshold
        
        if len(mixed_topics) > 1:
            return {
                'coherent': False,
                'reason': f'Multiple topics mixed: {", ".join(mixed_topics)}',
                'dominant_topic': dominant_topic,
                'coverage': coverage,
                'mixed_topics': mixed_topics,
                'requires_split': True
            }
        
        if not is_coherent:
            return {
                'coherent': False,
                'reason': f'Low topic coverage ({coverage:.0%}) - papers too diverse',
                'dominant_topic': dominant_topic,
                'coverage': coverage,
                'requires_split': True
            }
        
        return {
            'coherent': True,
            'dominant_topic': dominant_topic,
            'coverage': coverage,
            'requires_split': False
        }
    
    def calculate_title_similarity(self, df: pd.DataFrame) -> float:
        """
        Calculate how similar the titles are (simple word overlap)
        """
        if 'title' not in df.columns or len(df) < 2:
            return 0.0
        
        titles = df['title'].fillna('').astype(str).tolist()
        
        # Tokenize titles (simple word splitting)
        word_sets = []
        for title in titles:
            words = set(re.findall(r'\b\w+\b', title.lower()))
            # Remove common words
            common_words = {'a', 'an', 'the', 'in', 'on', 'at', 'for', 'with', 'and', 'or', 'of', 'to'}
            words = words - common_words
            if words:
                word_sets.append(words)
        
        if len(word_sets) < 2:
            return 0.0
        
        # Calculate average pairwise Jaccard similarity
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        return np.mean(similarities) if similarities else 0.0
    
    def analyze_cluster_quality(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        cluster_id: int
    ) -> Dict:
        """
        Comprehensive quality analysis of a cluster
        """
        
        cluster_mask = labels == cluster_id
        cluster_df = df[cluster_mask]
        
        if len(cluster_df) < 2:
            return {
                'cluster_id': cluster_id,
                'size': len(cluster_df),
                'quality': 'too_small',
                'should_keep': False
            }
        
        # Thematic coherence
        coherence = self.validate_cluster_coherence(cluster_df)
        
        # Title similarity
        title_sim = self.calculate_title_similarity(cluster_df)
        
        # Year span (should be reasonable)
        if 'publication_year' in cluster_df.columns:
            years = pd.to_numeric(cluster_df['publication_year'], errors='coerce').dropna()
            year_span = years.max() - years.min() if len(years) > 1 else 0
        else:
            year_span = 0
        
        # Overall quality assessment
        quality_score = 0.0
        reasons = []
        
        if coherence['coherent']:
            quality_score += 0.5
            reasons.append(f"✓ Coherent topic: {coherence['dominant_topic']} ({coherence['coverage']:.0%})")
        else:
            reasons.append(f"✗ {coherence['reason']}")
        
        if title_sim >= 0.2:  # At least 20% word overlap
            quality_score += 0.3
            reasons.append(f"✓ Similar titles ({title_sim:.0%} overlap)")
        else:
            reasons.append(f"✗ Titles too different ({title_sim:.0%} overlap)")
        
        if year_span <= 10:  # Within 10 years
            quality_score += 0.2
            reasons.append(f"✓ Reasonable time span ({year_span} years)")
        else:
            reasons.append(f"✗ Large time span ({year_span} years)")
        
        # Determine if we should keep this cluster
        should_keep = (
            coherence['coherent'] and 
            quality_score >= 0.5 and
            len(cluster_df) >= 5
        )
        
        return {
            'cluster_id': cluster_id,
            'size': len(cluster_df),
            'coherent': coherence['coherent'],
            'dominant_topic': coherence.get('dominant_topic'),
            'topic_coverage': coherence.get('coverage', 0),
            'title_similarity': title_sim,
            'year_span': year_span,
            'quality_score': quality_score,
            'should_keep': should_keep,
            'reasons': reasons,
            'requires_split': coherence.get('requires_split', False),
            'mixed_topics': coherence.get('mixed_topics', [])
        }
    
    def filter_incoherent_clusters(
        self,
        df: pd.DataFrame,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Filter out clusters that are not thematically coherent
        
        Returns:
            new_labels: Updated labels with incoherent clusters marked as noise
            report: Analysis report
        """
        
        new_labels = labels.copy()
        cluster_analyses = {}
        
        unique_clusters = [c for c in np.unique(labels) if c != -1]
        
        kept_clusters = []
        removed_clusters = []
        
        for cluster_id in unique_clusters:
            analysis = self.analyze_cluster_quality(df, labels, cluster_id)
            cluster_analyses[cluster_id] = analysis
            
            if not analysis['should_keep']:
                # Mark as noise
                cluster_mask = labels == cluster_id
                new_labels[cluster_mask] = -1
                removed_clusters.append(cluster_id)
            else:
                kept_clusters.append(cluster_id)
        
        # Renumber remaining clusters
        next_id = 0
        id_mapping = {}
        for old_id in kept_clusters:
            id_mapping[old_id] = next_id
            next_id += 1
        
        for old_id, new_id in id_mapping.items():
            mask = labels == old_id
            new_labels[mask] = new_id
        
        report = {
            'original_clusters': len(unique_clusters),
            'kept_clusters': len(kept_clusters),
            'removed_clusters': len(removed_clusters),
            'cluster_analyses': cluster_analyses,
            'removed_ids': removed_clusters
        }
        
        return new_labels, report
