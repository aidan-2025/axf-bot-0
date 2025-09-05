#!/usr/bin/env python3
"""
Ensemble Sentiment Analyzer
Combines multiple sentiment analysis methods for robust results
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from statistics import mean, median

from .models import SentimentResult, SentimentLabel, SentimentSource, SentimentBatchResult
from .bert_sentiment_analyzer import BERTSentimentAnalyzer, BERTConfig
from .lexicon_sentiment_analyzer import LexiconSentimentAnalyzer, LexiconConfig

logger = logging.getLogger(__name__)

@dataclass
class EnsembleConfig:
    """Configuration for ensemble sentiment analyzer"""
    # Model weights (should sum to 1.0)
    bert_weight: float = 0.6
    lexicon_weight: float = 0.4
    
    # BERT configuration
    bert_config: Optional[BERTConfig] = None
    
    # Lexicon configuration
    lexicon_config: Optional[LexiconConfig] = None
    
    # Ensemble settings
    confidence_threshold: float = 0.7
    enable_agreement_boost: bool = True
    agreement_threshold: float = 0.8
    fallback_to_lexicon: bool = True

class EnsembleSentimentAnalyzer:
    """Ensemble sentiment analyzer combining multiple methods"""
    
    def __init__(self, config: EnsembleConfig):
        """Initialize ensemble sentiment analyzer"""
        self.config = config
        
        # Initialize analyzers
        self.bert_analyzer = BERTSentimentAnalyzer(
            config.bert_config or BERTConfig()
        )
        self.lexicon_analyzer = LexiconSentimentAnalyzer(
            config.lexicon_config or LexiconConfig()
        )
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all analyzers"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing ensemble sentiment analyzer")
            
            # Initialize BERT analyzer
            await self.bert_analyzer.initialize()
            
            # Initialize lexicon analyzer
            await self.lexicon_analyzer.initialize()
            
            self._initialized = True
            logger.info("Ensemble sentiment analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ensemble analyzer: {e}")
            if self.config.fallback_to_lexicon:
                logger.info("Falling back to lexicon-only mode")
                self._initialized = True
            else:
                raise
    
    async def analyze_text(self, 
                          text: str,
                          language: str = "en",
                          currency_pairs: Optional[List[str]] = None) -> SentimentResult:
        """Analyze sentiment using ensemble method"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Get results from all analyzers
            results = []
            
            # BERT analysis
            try:
                bert_result = await self.bert_analyzer.analyze_text(text, language, currency_pairs)
                results.append(('bert', bert_result))
            except Exception as e:
                logger.warning(f"BERT analysis failed: {e}")
                if not self.config.fallback_to_lexicon:
                    raise
            
            # Lexicon analysis
            try:
                lexicon_result = await self.lexicon_analyzer.analyze_text(text, language, currency_pairs)
                results.append(('lexicon', lexicon_result))
            except Exception as e:
                logger.warning(f"Lexicon analysis failed: {e}")
                if not results:  # No other results available
                    raise
            
            # Combine results
            ensemble_result = self._combine_results(results, text, language, currency_pairs)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            ensemble_result.processing_time_ms = processing_time
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error in ensemble analysis: {e}")
            # Return neutral sentiment as fallback
            return SentimentResult(
                text=text,
                label=SentimentLabel.NEUTRAL,
                score=0.0,
                confidence=0.0,
                source=SentimentSource.ENSEMBLE,
                language=language,
                processing_time_ms=(time.time() - start_time) * 1000,
                currency_pairs=currency_pairs or []
            )
    
    async def analyze_batch(self, 
                           texts: List[str],
                           languages: Optional[List[str]] = None,
                           currency_pairs_list: Optional[List[List[str]]] = None) -> List[SentimentResult]:
        """Analyze sentiment of multiple texts using ensemble method"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        results = []
        
        try:
            # Get results from all analyzers
            bert_results = []
            lexicon_results = []
            
            # BERT analysis
            try:
                bert_results = await self.bert_analyzer.analyze_batch(
                    texts, languages, currency_pairs_list
                )
            except Exception as e:
                logger.warning(f"BERT batch analysis failed: {e}")
                if not self.config.fallback_to_lexicon:
                    raise
            
            # Lexicon analysis
            try:
                lexicon_results = await self.lexicon_analyzer.analyze_batch(
                    texts, languages, currency_pairs_list
                )
            except Exception as e:
                logger.warning(f"Lexicon batch analysis failed: {e}")
                if not bert_results:  # No other results available
                    raise
            
            # Combine results for each text
            for i, text in enumerate(texts):
                text_results = []
                
                if i < len(bert_results):
                    text_results.append(('bert', bert_results[i]))
                if i < len(lexicon_results):
                    text_results.append(('lexicon', lexicon_results[i]))
                
                if text_results:
                    language = (languages or ["en"] * len(texts))[i]
                    currency_pairs = (currency_pairs_list or [[]] * len(texts))[i]
                    
                    ensemble_result = self._combine_results(text_results, text, language, currency_pairs)
                    results.append(ensemble_result)
                else:
                    # Fallback neutral result
                    results.append(SentimentResult(
                        text=text,
                        label=SentimentLabel.NEUTRAL,
                        score=0.0,
                        confidence=0.0,
                        source=SentimentSource.ENSEMBLE,
                        language=(languages or ["en"] * len(texts))[i],
                        processing_time_ms=0.0,
                        currency_pairs=(currency_pairs_list or [[]] * len(texts))[i]
                    ))
            
            total_time = (time.time() - start_time) * 1000
            logger.info(f"Analyzed {len(texts)} texts with ensemble in {total_time:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Error in ensemble batch analysis: {e}")
            # Return neutral sentiments as fallback
            return [
                SentimentResult(
                    text=text,
                    label=SentimentLabel.NEUTRAL,
                    score=0.0,
                    confidence=0.0,
                    source=SentimentSource.ENSEMBLE,
                    language=(languages or ["en"] * len(texts))[i],
                    processing_time_ms=0.0,
                    currency_pairs=(currency_pairs_list or [[]] * len(texts))[i]
                )
                for i, text in enumerate(texts)
            ]
    
    def _combine_results(self, 
                        results: List[tuple[str, SentimentResult]], 
                        text: str,
                        language: str,
                        currency_pairs: Optional[List[str]]) -> SentimentResult:
        """Combine results from multiple analyzers"""
        if not results:
            return SentimentResult(
                text=text,
                label=SentimentLabel.NEUTRAL,
                score=0.0,
                confidence=0.0,
                source=SentimentSource.ENSEMBLE,
                language=language,
                currency_pairs=currency_pairs or []
            )
        
        # Extract scores and confidences
        scores = []
        confidences = []
        labels = []
        weights = []
        
        for analyzer_name, result in results:
            scores.append(result.score)
            confidences.append(result.confidence)
            labels.append(result.label)
            
            # Get weight for this analyzer
            if analyzer_name == 'bert':
                weights.append(self.config.bert_weight)
            elif analyzer_name == 'lexicon':
                weights.append(self.config.lexicon_weight)
            else:
                weights.append(0.1)  # Default weight for unknown analyzers
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # Calculate weighted average score
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        
        # Calculate weighted average confidence
        weighted_confidence = sum(conf * weight for conf, weight in zip(confidences, weights))
        
        # Check for agreement boost
        if self.config.enable_agreement_boost and len(results) > 1:
            agreement = self._calculate_agreement(labels, scores)
            if agreement >= self.config.agreement_threshold:
                weighted_confidence = min(1.0, weighted_confidence * 1.2)  # 20% boost
        
        # Determine final label
        final_label = self._determine_ensemble_label(weighted_score, weighted_confidence)
        
        # Combine financial entities
        all_entities = set()
        for _, result in results:
            all_entities.update(result.financial_entities)
        
        # Create ensemble result
        ensemble_result = SentimentResult(
            text=text,
            label=final_label,
            score=weighted_score,
            confidence=weighted_confidence,
            source=SentimentSource.ENSEMBLE,
            language=language,
            processing_time_ms=None,  # Will be set by caller
            model_name="ensemble",
            model_version="1.0",
            currency_pairs=currency_pairs or [],
            financial_entities=list(all_entities)
        )
        
        return ensemble_result
    
    def _calculate_agreement(self, labels: List[SentimentLabel], scores: List[float]) -> float:
        """Calculate agreement between analyzers"""
        if len(labels) < 2:
            return 1.0
        
        # Check label agreement
        label_agreement = len(set(labels)) == 1
        
        # Check score agreement (same direction)
        positive_scores = sum(1 for score in scores if score > 0)
        negative_scores = sum(1 for score in scores if score < 0)
        neutral_scores = sum(1 for score in scores if score == 0)
        
        score_agreement = max(positive_scores, negative_scores, neutral_scores) / len(scores)
        
        # Combine agreements
        return (label_agreement + score_agreement) / 2
    
    def _determine_ensemble_label(self, score: float, confidence: float) -> SentimentLabel:
        """Determine final sentiment label from ensemble score"""
        abs_score = abs(score)
        
        # Apply confidence threshold
        if confidence < self.config.confidence_threshold:
            return SentimentLabel.NEUTRAL
        
        if abs_score < 0.1:
            return SentimentLabel.NEUTRAL
        elif score > 0:
            if abs_score > 0.5:
                return SentimentLabel.VERY_POSITIVE
            else:
                return SentimentLabel.POSITIVE
        else:
            if abs_score > 0.5:
                return SentimentLabel.VERY_NEGATIVE
            else:
                return SentimentLabel.NEGATIVE
    
    async def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about the ensemble analyzer"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        info = {
            "status": "initialized",
            "bert_weight": self.config.bert_weight,
            "lexicon_weight": self.config.lexicon_weight,
            "confidence_threshold": self.config.confidence_threshold,
            "enable_agreement_boost": self.config.enable_agreement_boost,
            "fallback_to_lexicon": self.config.fallback_to_lexicon
        }
        
        # Add analyzer info
        try:
            bert_info = await self.bert_analyzer.get_model_info()
            info["bert_analyzer"] = bert_info
        except Exception as e:
            info["bert_analyzer"] = {"error": str(e)}
        
        try:
            lexicon_info = await self.lexicon_analyzer.get_lexicon_info()
            info["lexicon_analyzer"] = lexicon_info
        except Exception as e:
            info["lexicon_analyzer"] = {"error": str(e)}
        
        return info
    
    async def close(self):
        """Close all analyzers"""
        try:
            await self.bert_analyzer.close()
        except Exception as e:
            logger.warning(f"Error closing BERT analyzer: {e}")
        
        self._initialized = False
        logger.info("Ensemble analyzer closed")

