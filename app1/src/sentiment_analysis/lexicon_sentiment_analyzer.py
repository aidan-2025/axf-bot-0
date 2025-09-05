#!/usr/bin/env python3
"""
Lexicon Sentiment Analyzer
Rule-based sentiment analysis using financial lexicons
"""

import asyncio
import logging
import re
import json
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path

from .models import SentimentResult, SentimentLabel, SentimentSource

logger = logging.getLogger(__name__)

@dataclass
class LexiconConfig:
    """Configuration for lexicon sentiment analyzer"""
    lexicon_path: Optional[str] = None
    enable_negation: bool = True
    enable_intensifiers: bool = True
    enable_diminishers: bool = True
    min_word_length: int = 2
    confidence_threshold: float = 0.5

class LexiconSentimentAnalyzer:
    """Rule-based sentiment analyzer using financial lexicons"""
    
    def __init__(self, config: LexiconConfig):
        """Initialize lexicon sentiment analyzer"""
        self.config = config
        self.positive_words: Set[str] = set()
        self.negative_words: Set[str] = set()
        self.intensifiers: Set[str] = set()
        self.diminishers: Set[str] = set()
        self.negation_words: Set[str] = set()
        self._initialized = False
        
    async def initialize(self):
        """Initialize the lexicon analyzer"""
        if self._initialized:
            return
            
        try:
            logger.info("Initializing lexicon sentiment analyzer")
            
            # Load default lexicons
            await self._load_default_lexicons()
            
            # Load custom lexicon if provided
            if self.config.lexicon_path:
                await self._load_custom_lexicon()
            
            self._initialized = True
            logger.info(f"Lexicon analyzer initialized with {len(self.positive_words)} positive and {len(self.negative_words)} negative words")
            
        except Exception as e:
            logger.error(f"Failed to initialize lexicon analyzer: {e}")
            raise
    
    async def _load_default_lexicons(self):
        """Load default financial sentiment lexicons"""
        # Financial positive words
        self.positive_words = {
            # Market performance
            'bullish', 'rally', 'surge', 'soar', 'climb', 'rise', 'gain', 'increase',
            'growth', 'expansion', 'boom', 'prosperity', 'recovery', 'rebound',
            'outperform', 'beat', 'exceed', 'strong', 'robust', 'solid', 'healthy',
            'positive', 'optimistic', 'confident', 'upbeat', 'favorable',
            
            # Economic indicators
            'inflation', 'deflation', 'stability', 'equilibrium', 'balance',
            'efficiency', 'productivity', 'innovation', 'development', 'progress',
            'success', 'achievement', 'milestone', 'breakthrough', 'advance',
            
            # Market sentiment
            'buy', 'purchase', 'invest', 'acquisition', 'merger', 'partnership',
            'collaboration', 'agreement', 'deal', 'contract', 'settlement',
            'resolution', 'solution', 'improvement', 'enhancement', 'upgrade',
            
            # Financial terms
            'profit', 'earnings', 'revenue', 'income', 'dividend', 'yield',
            'return', 'performance', 'outlook', 'forecast', 'projection',
            'target', 'goal', 'objective', 'strategy', 'plan', 'initiative'
        }
        
        # Financial negative words
        self.negative_words = {
            # Market performance
            'bearish', 'crash', 'plunge', 'fall', 'drop', 'decline', 'decrease',
            'loss', 'recession', 'depression', 'crisis', 'collapse', 'bust',
            'underperform', 'miss', 'disappoint', 'weak', 'fragile', 'vulnerable',
            'negative', 'pessimistic', 'concerned', 'worried', 'cautious',
            
            # Economic indicators
            'inflation', 'deflation', 'instability', 'volatility', 'uncertainty',
            'risk', 'threat', 'challenge', 'problem', 'issue', 'difficulty',
            'failure', 'setback', 'decline', 'deterioration', 'downturn',
            
            # Market sentiment
            'sell', 'dump', 'exit', 'withdraw', 'divest', 'disposal', 'liquidation',
            'bankruptcy', 'default', 'insolvency', 'restructuring', 'layoff',
            'cut', 'reduction', 'decrease', 'downgrade', 'downturn',
            
            # Financial terms
            'loss', 'deficit', 'debt', 'liability', 'expense', 'cost', 'burden',
            'pressure', 'stress', 'strain', 'tension', 'conflict', 'dispute',
            'litigation', 'lawsuit', 'penalty', 'fine', 'sanction', 'restriction'
        }
        
        # Intensifiers (amplify sentiment)
        if self.config.enable_intensifiers:
            self.intensifiers = {
                'very', 'extremely', 'highly', 'significantly', 'substantially',
                'dramatically', 'considerably', 'tremendously', 'massively',
                'completely', 'totally', 'absolutely', 'entirely', 'fully',
                'greatly', 'vastly', 'immensely', 'enormously', 'hugely'
            }
        
        # Diminishers (reduce sentiment)
        if self.config.enable_diminishers:
            self.diminishers = {
                'slightly', 'somewhat', 'partially', 'moderately', 'relatively',
                'fairly', 'reasonably', 'somewhat', 'a bit', 'a little',
                'marginally', 'minimally', 'barely', 'hardly', 'scarcely'
            }
        
        # Negation words
        if self.config.enable_negation:
            self.negation_words = {
                'not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere',
                'neither', 'nor', 'cannot', 'can\'t', 'won\'t', 'don\'t',
                'doesn\'t', 'didn\'t', 'haven\'t', 'hasn\'t', 'hadn\'t',
                'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'without',
                'lack', 'absence', 'missing', 'devoid', 'free'
            }
    
    async def _load_custom_lexicon(self):
        """Load custom lexicon from file"""
        try:
            lexicon_path = Path(self.config.lexicon_path)
            if not lexicon_path.exists():
                logger.warning(f"Custom lexicon file not found: {lexicon_path}")
                return
            
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                lexicon_data = json.load(f)
            
            # Load custom words
            if 'positive' in lexicon_data:
                self.positive_words.update(lexicon_data['positive'])
            if 'negative' in lexicon_data:
                self.negative_words.update(lexicon_data['negative'])
            if 'intensifiers' in lexicon_data:
                self.intensifiers.update(lexicon_data['intensifiers'])
            if 'diminishers' in lexicon_data:
                self.diminishers.update(lexicon_data['diminishers'])
            if 'negation' in lexicon_data:
                self.negation_words.update(lexicon_data['negation'])
            
            logger.info("Custom lexicon loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load custom lexicon: {e}")
    
    async def analyze_text(self, 
                          text: str,
                          language: str = "en",
                          currency_pairs: Optional[List[str]] = None) -> SentimentResult:
        """Analyze sentiment of a single text"""
        if not self._initialized:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Preprocess text
            words = self._preprocess_text(text)
            
            # Calculate sentiment score
            score = self._calculate_sentiment_score(words)
            
            # Determine label and confidence
            label, confidence = self._determine_label_and_confidence(score)
            
            # Calculate processing time
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Create result
            result = SentimentResult(
                text=text,
                label=label,
                score=score,
                confidence=confidence,
                source=SentimentSource.LEXICON,
                language=language,
                processing_time_ms=processing_time,
                model_name="financial_lexicon",
                currency_pairs=currency_pairs or [],
                financial_entities=self._extract_financial_entities(text)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text with lexicon: {e}")
            # Return neutral sentiment as fallback
            return SentimentResult(
                text=text,
                label=SentimentLabel.NEUTRAL,
                score=0.0,
                confidence=0.0,
                source=SentimentSource.LEXICON,
                language=language,
                processing_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000,
                model_name="financial_lexicon",
                currency_pairs=currency_pairs or []
            )
    
    async def analyze_batch(self, 
                           texts: List[str],
                           languages: Optional[List[str]] = None,
                           currency_pairs_list: Optional[List[List[str]]] = None) -> List[SentimentResult]:
        """Analyze sentiment of multiple texts"""
        if not self._initialized:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        results = []
        
        try:
            for i, text in enumerate(texts):
                language = (languages or ["en"] * len(texts))[i]
                currency_pairs = (currency_pairs_list or [[]] * len(texts))[i]
                
                result = await self.analyze_text(text, language, currency_pairs)
                results.append(result)
            
            total_time = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.info(f"Analyzed {len(texts)} texts with lexicon in {total_time:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing batch with lexicon: {e}")
            # Return neutral sentiments as fallback
            return [
                SentimentResult(
                    text=text,
                    label=SentimentLabel.NEUTRAL,
                    score=0.0,
                    confidence=0.0,
                    source=SentimentSource.LEXICON,
                    language=(languages or ["en"] * len(texts))[i],
                    processing_time_ms=0.0,
                    model_name="financial_lexicon",
                    currency_pairs=(currency_pairs_list or [[]] * len(texts))[i]
                )
                for i, text in enumerate(texts)
            ]
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for lexicon analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text)
        
        # Filter by minimum length
        words = [word for word in words if len(word) >= self.config.min_word_length]
        
        return words
    
    def _calculate_sentiment_score(self, words: List[str]) -> float:
        """Calculate sentiment score for words"""
        score = 0.0
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        for i, word in enumerate(words):
            word_score = 0.0
            
            # Check for positive/negative words
            if word in self.positive_words:
                word_score = 1.0
            elif word in self.negative_words:
                word_score = -1.0
            
            if word_score != 0.0:
                # Check for negation
                if self._is_negated(word, words, i):
                    word_score = -word_score
                
                # Check for intensifiers/diminishers
                multiplier = self._get_intensity_multiplier(word, words, i)
                word_score *= multiplier
                
                score += word_score
        
        # Normalize score
        return score / total_words
    
    def _is_negated(self, word: str, words: List[str], position: int) -> bool:
        """Check if word is negated"""
        if not self.config.enable_negation:
            return False
        
        # Check previous words for negation
        for i in range(max(0, position - 3), position):
            if words[i] in self.negation_words:
                return True
        
        return False
    
    def _get_intensity_multiplier(self, word: str, words: List[str], position: int) -> float:
        """Get intensity multiplier for word"""
        multiplier = 1.0
        
        # Check for intensifiers
        if self.config.enable_intensifiers:
            for i in range(max(0, position - 2), position):
                if words[i] in self.intensifiers:
                    multiplier *= 1.5
                    break
        
        # Check for diminishers
        if self.config.enable_diminishers:
            for i in range(max(0, position - 2), position):
                if words[i] in self.diminishers:
                    multiplier *= 0.5
                    break
        
        return multiplier
    
    def _determine_label_and_confidence(self, score: float) -> tuple[SentimentLabel, float]:
        """Determine sentiment label and confidence from score"""
        abs_score = abs(score)
        
        if abs_score < 0.1:
            label = SentimentLabel.NEUTRAL
            confidence = max(0.1, abs_score)
        elif score > 0:
            if abs_score > 0.5:
                label = SentimentLabel.VERY_POSITIVE
            else:
                label = SentimentLabel.POSITIVE
            confidence = min(1.0, abs_score)
        else:
            if abs_score > 0.5:
                label = SentimentLabel.VERY_NEGATIVE
            else:
                label = SentimentLabel.NEGATIVE
            confidence = min(1.0, abs_score)
        
        # Apply confidence threshold
        if confidence < self.config.confidence_threshold:
            label = SentimentLabel.NEUTRAL
            confidence = 0.1
        
        return label, confidence
    
    def _extract_financial_entities(self, text: str) -> List[str]:
        """Extract financial entities from text"""
        entities = []
        
        # Currency pairs
        currency_pattern = r'\b[A-Z]{3}/[A-Z]{3}\b'
        entities.extend(re.findall(currency_pattern, text.upper()))
        
        # Financial terms
        financial_terms = [
            'GDP', 'CPI', 'inflation', 'interest rate', 'unemployment',
            'retail sales', 'manufacturing', 'trade balance', 'fiscal',
            'monetary policy', 'central bank', 'fed', 'ecb', 'boe'
        ]
        
        text_lower = text.lower()
        for term in financial_terms:
            if term in text_lower:
                entities.append(term.upper())
        
        return list(set(entities))
    
    async def get_lexicon_info(self) -> Dict[str, Any]:
        """Get information about the loaded lexicon"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "positive_words": len(self.positive_words),
            "negative_words": len(self.negative_words),
            "intensifiers": len(self.intensifiers),
            "diminishers": len(self.diminishers),
            "negation_words": len(self.negation_words),
            "enable_negation": self.config.enable_negation,
            "enable_intensifiers": self.config.enable_intensifiers,
            "enable_diminishers": self.config.enable_diminishers
        }

