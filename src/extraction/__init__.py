"""Extraction modules"""
from .gap_analyzer import GapAnalyzer
from .hypothesis_generator import HypothesisGenerator
from .classification_validator import ClassificationValidator
from .custom_criteria import (
    CustomCriteriaValidator,
    ValidationCriterion,
    ClinicalTrialSponsorCriterion,
    DataAvailabilityCriterion,
    ReplicationStatusCriterion
)

__all__ = [
    'GapAnalyzer', 
    'HypothesisGenerator', 
    'ClassificationValidator',
    'CustomCriteriaValidator',
    'ValidationCriterion',
    'ClinicalTrialSponsorCriterion',
    'DataAvailabilityCriterion',
    'ReplicationStatusCriterion'
]
