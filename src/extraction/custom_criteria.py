"""
Custom Criteria Module
Extensible framework for adding custom validation criteria
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any
import re
from abc import ABC, abstractmethod


class ValidationCriterion(ABC):
    """
    Abstract base class for validation criteria
    All custom criteria should inherit from this
    """
    
    def __init__(self, name: str, weight: float = 0.1):
        """
        Args:
            name: Criterion name
            weight: Weight in overall validation score (0-1)
        """
        self.name = name
        self.weight = weight
    
    @abstractmethod
    def evaluate(self, cluster_df: pd.DataFrame, text_column: str = 'abstract') -> Dict:
        """
        Evaluate criterion for a cluster
        
        Returns:
            Dict with:
                - score: float (0-1)
                - details: Dict with analysis details
                - interpretation: str explaining the score
        """
        pass
    
    def get_weight(self) -> float:
        """Get criterion weight"""
        return self.weight


class ClinicalTrialSponsorCriterion(ValidationCriterion):
    """
    Validates if clinical trials have sponsor information
    Useful for identifying industry vs academic research
    """
    
    def __init__(self, weight: float = 0.15):
        super().__init__("clinical_trial_sponsor", weight)
        
        # Keywords indicating clinical trials
        self.trial_keywords = [
            'clinical trial', 'randomized', 'rct', 'phase i', 'phase ii',
            'phase iii', 'phase iv', 'controlled trial', 'double-blind'
        ]
        
        # Keywords indicating sponsors
        self.sponsor_keywords = [
            'funded by', 'sponsored by', 'support', 'grant',
            'pharmaceutical', 'biotech', 'industry', 'company',
            'corporation', 'inc.', 'ltd.', 'llc'
        ]
        
        # Academic/non-profit indicators
        self.academic_keywords = [
            'nih', 'nsf', 'university', 'institute', 'foundation',
            'government', 'public health', 'national'
        ]
    
    def evaluate(self, cluster_df: pd.DataFrame, text_column: str = 'abstract') -> Dict:
        """Evaluate sponsor information in clinical trials"""
        
        # Combine all text
        all_text = ' '.join(
            cluster_df[text_column].dropna().astype(str).str.lower()
        )
        
        # Check if this is a clinical trial cluster
        trial_mentions = sum(
            1 for keyword in self.trial_keywords 
            if keyword in all_text
        )
        
        is_trial_cluster = trial_mentions >= 3
        
        if not is_trial_cluster:
            # Not a clinical trial cluster - neutral score
            return {
                'score': 0.7,
                'details': {
                    'is_clinical_trial_cluster': False,
                    'trial_mentions': trial_mentions
                },
                'interpretation': 'Not a clinical trial cluster - criterion not applicable'
            }
        
        # Count sponsor mentions
        sponsor_mentions = sum(
            1 for keyword in self.sponsor_keywords 
            if keyword in all_text
        )
        
        academic_mentions = sum(
            1 for keyword in self.academic_keywords 
            if keyword in all_text
        )
        
        # Calculate sponsor coverage
        n_papers = len(cluster_df)
        sponsor_coverage = sponsor_mentions / n_papers if n_papers > 0 else 0
        
        # Score based on sponsor information presence
        if sponsor_coverage >= 0.5:
            score = 1.0  # Good - most trials have sponsor info
        elif sponsor_coverage >= 0.3:
            score = 0.8  # Acceptable
        elif sponsor_coverage >= 0.1:
            score = 0.6  # Some info
        else:
            score = 0.4  # Poor - missing sponsor info
        
        # Determine funding type
        if academic_mentions > sponsor_mentions * 2:
            funding_type = 'primarily_academic'
        elif sponsor_mentions > academic_mentions * 2:
            funding_type = 'primarily_industry'
        else:
            funding_type = 'mixed'
        
        return {
            'score': score,
            'details': {
                'is_clinical_trial_cluster': True,
                'trial_mentions': trial_mentions,
                'sponsor_mentions': sponsor_mentions,
                'academic_mentions': academic_mentions,
                'sponsor_coverage': sponsor_coverage,
                'funding_type': funding_type,
                'n_papers': n_papers
            },
            'interpretation': self._generate_interpretation(
                sponsor_coverage, funding_type
            )
        }
    
    def _generate_interpretation(self, coverage: float, funding_type: str) -> str:
        """Generate human-readable interpretation"""
        
        if coverage >= 0.5:
            coverage_desc = "Strong sponsor information"
        elif coverage >= 0.3:
            coverage_desc = "Moderate sponsor information"
        else:
            coverage_desc = "Limited sponsor information"
        
        funding_desc = {
            'primarily_academic': 'primarily academic/government funded',
            'primarily_industry': 'primarily industry sponsored',
            'mixed': 'mixed funding sources'
        }.get(funding_type, 'unknown funding')
        
        return f"{coverage_desc}. Clinical trials appear {funding_desc}."


class DataAvailabilityCriterion(ValidationCriterion):
    """
    Checks if papers mention data availability
    Important for reproducibility and hypothesis verification
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__("data_availability", weight)
        
        self.availability_keywords = [
            'data available', 'publicly available', 'open data',
            'data repository', 'github', 'figshare', 'zenodo',
            'supplementary data', 'available upon request',
            'data sharing', 'open access'
        ]
    
    def evaluate(self, cluster_df: pd.DataFrame, text_column: str = 'abstract') -> Dict:
        """Evaluate data availability mentions"""
        
        all_text = ' '.join(
            cluster_df[text_column].dropna().astype(str).str.lower()
        )
        
        availability_mentions = sum(
            1 for keyword in self.availability_keywords 
            if keyword in all_text
        )
        
        n_papers = len(cluster_df)
        availability_rate = availability_mentions / n_papers if n_papers > 0 else 0
        
        # Score based on data availability mentions
        if availability_rate >= 0.5:
            score = 1.0
            interpretation = "High data availability - good for hypothesis verification"
        elif availability_rate >= 0.3:
            score = 0.8
            interpretation = "Moderate data availability"
        elif availability_rate >= 0.1:
            score = 0.6
            interpretation = "Some data availability mentioned"
        else:
            score = 0.5
            interpretation = "Limited data availability information"
        
        return {
            'score': score,
            'details': {
                'availability_mentions': availability_mentions,
                'availability_rate': availability_rate,
                'n_papers': n_papers
            },
            'interpretation': interpretation
        }


class ReplicationStatusCriterion(ValidationCriterion):
    """
    Identifies if papers are replications or original studies
    Important for understanding research maturity
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__("replication_status", weight)
        
        self.replication_keywords = [
            'replication', 'replicate', 'reproduce', 'validation study',
            'confirmatory', 'verify', 'independent validation'
        ]
        
        self.original_keywords = [
            'novel', 'first', 'new', 'original', 'discovery',
            'unprecedented', 'innovative'
        ]
    
    def evaluate(self, cluster_df: pd.DataFrame, text_column: str = 'abstract') -> Dict:
        """Evaluate replication vs original research"""
        
        all_text = ' '.join(
            cluster_df[text_column].dropna().astype(str).str.lower()
        )
        
        replication_mentions = sum(
            1 for keyword in self.replication_keywords 
            if keyword in all_text
        )
        
        original_mentions = sum(
            1 for keyword in self.original_keywords 
            if keyword in all_text
        )
        
        total_mentions = replication_mentions + original_mentions
        
        if total_mentions == 0:
            research_type = 'unclear'
            score = 0.7
        elif replication_mentions > original_mentions:
            research_type = 'replication_focused'
            score = 0.9  # Replications are valuable
        elif original_mentions > replication_mentions * 2:
            research_type = 'original_research'
            score = 0.8
        else:
            research_type = 'mixed'
            score = 0.85
        
        return {
            'score': score,
            'details': {
                'replication_mentions': replication_mentions,
                'original_mentions': original_mentions,
                'research_type': research_type
            },
            'interpretation': f"Cluster appears to be {research_type.replace('_', ' ')}"
        }


class CustomCriteriaValidator:
    """
    Extensible validator that can incorporate custom criteria
    """
    
    def __init__(self):
        self.criteria: List[ValidationCriterion] = []
        self.results = {}
    
    def add_criterion(self, criterion: ValidationCriterion):
        """Add a custom validation criterion"""
        self.criteria.append(criterion)
        print(f"✅ Added criterion: {criterion.name} (weight: {criterion.weight})")
    
    def remove_criterion(self, name: str):
        """Remove a criterion by name"""
        self.criteria = [c for c in self.criteria if c.name != name]
        print(f"🗑️  Removed criterion: {name}")
    
    def list_criteria(self):
        """List all active criteria"""
        print("\n📋 Active Validation Criteria:")
        print("=" * 60)
        for criterion in self.criteria:
            print(f"  • {criterion.name} (weight: {criterion.weight})")
        print("=" * 60)
    
    def evaluate_cluster(
        self,
        cluster_df: pd.DataFrame,
        cluster_id: int,
        text_column: str = 'abstract'
    ) -> Dict:
        """
        Evaluate cluster using all custom criteria
        
        Returns:
            Dict with scores and details for each criterion
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING CLUSTER {cluster_id} WITH CUSTOM CRITERIA")
        print(f"{'='*70}")
        
        results = {
            'cluster_id': cluster_id,
            'size': len(cluster_df),
            'criteria_results': {},
            'overall_custom_score': 0.0
        }
        
        total_weight = sum(c.weight for c in self.criteria)
        weighted_score = 0.0
        
        for criterion in self.criteria:
            print(f"\n🔍 Evaluating: {criterion.name}")
            
            result = criterion.evaluate(cluster_df, text_column)
            results['criteria_results'][criterion.name] = result
            
            weighted_score += result['score'] * criterion.weight
            
            print(f"   Score: {result['score']:.2f}")
            print(f"   {result['interpretation']}")
        
        if total_weight > 0:
            results['overall_custom_score'] = weighted_score / total_weight
        
        print(f"\n📊 Overall Custom Score: {results['overall_custom_score']:.2f}")
        print(f"{'='*70}\n")
        
        return results
    
    def evaluate_all_clusters(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        text_column: str = 'abstract'
    ) -> Dict:
        """Evaluate all clusters with custom criteria"""
        
        print("\n" + "="*70)
        print("CUSTOM CRITERIA VALIDATION")
        print("="*70)
        
        self.list_criteria()
        
        results = {}
        unique_clusters = [c for c in set(labels) if c != -1]
        
        for cluster_id in unique_clusters:
            mask = labels == cluster_id
            cluster_df = df[mask]
            
            cluster_results = self.evaluate_cluster(
                cluster_df, 
                cluster_id, 
                text_column
            )
            results[f"cluster_{cluster_id}"] = cluster_results
        
        return {
            'custom_criteria_results': results,
            'criteria_used': [c.name for c in self.criteria]
        }


# Example: Create a custom criterion
class CustomExampleCriterion(ValidationCriterion):
    """
    Template for creating custom criteria
    Copy and modify this for your specific needs
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__("custom_example", weight)
        
        # Define your keywords or patterns
        self.keywords = ['keyword1', 'keyword2']
    
    def evaluate(self, cluster_df: pd.DataFrame, text_column: str = 'abstract') -> Dict:
        """
        Implement your custom evaluation logic
        
        Must return dict with:
            - score: float (0-1)
            - details: dict
            - interpretation: str
        """
        
        # Your evaluation logic here
        all_text = ' '.join(
            cluster_df[text_column].dropna().astype(str).str.lower()
        )
        
        # Example: count keyword mentions
        mentions = sum(1 for kw in self.keywords if kw in all_text)
        score = min(mentions / 10, 1.0)  # Example scoring
        
        return {
            'score': score,
            'details': {
                'mentions': mentions,
                'n_papers': len(cluster_df)
            },
            'interpretation': f"Found {mentions} mentions of target keywords"
        }


if __name__ == "__main__":
    print("Custom Criteria Module")
    print("=" * 50)
    print("\nExtensible validation framework for custom criteria")
    print("\nBuilt-in criteria:")
    print("  • ClinicalTrialSponsorCriterion - Sponsor information")
    print("  • DataAvailabilityCriterion - Data sharing")
    print("  • ReplicationStatusCriterion - Original vs replication")
    print("\nCreate custom criteria by inheriting from ValidationCriterion")
