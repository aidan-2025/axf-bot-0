#!/usr/bin/env python3
"""
Sentiment Processor
Main processor for sentiment analysis with trend tracking
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from .models import (
    SentimentResult, SentimentLabel, SentimentSource, 
    SentimentTrend, SentimentBatchResult
)
from .ensemble_sentiment_analyzer import EnsembleSentimentAnalyzer, EnsembleConfig
from .sentiment_trend_analyzer import SentimentTrendAnalyzer, TrendConfig

logger = logging.getLogger(__name__)

@dataclass
class SentimentProcessorConfig:
    """Configuration for sentiment processor"""
    # Analyzer configuration
    ensemble_config: Optional[EnsembleConfig] = None
    
    # Trend analysis configuration
    trend_config: Optional[TrendConfig] = None
    
    # Processing settings
    batch_size: int = 50
    max_processing_time_ms: float = 5000.0
    enable_trend_analysis: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

class SentimentProcessor:
    """Main sentiment processor with trend analysis"""
    
    def __init__(self, config: SentimentProcessorConfig):
        """Initialize sentiment processor"""
        self.config = config
        
        # Initialize analyzers
        self.ensemble_analyzer = EnsembleSentimentAnalyzer(
            config.ensemble_config or EnsembleConfig()
        )
        
        self.trend_analyzer = SentimentTrendAnalyzer(
            config.trend_config or TrendConfig()
        ) if config.enable_trend_analysis else None
        
        # Cache for results
        self.cache: Dict[str, SentimentResult] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize the sentiment processor"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing sentiment processor")
            
            # Initialize ensemble analyzer
            await self.ensemble_analyzer.initialize()
            
            # Initialize trend analyzer
            if self.trend_analyzer:
                await self.trend_analyzer.initialize()
            
            self._initialized = True
            logger.info("Sentiment processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize sentiment processor: {e}")
            raise
    
    async def analyze_text(self, 
                          text: str,
                          language: str = "en",
                          currency_pairs: Optional[List[str]] = None,
                          use_cache: bool = True) -> SentimentResult:
        """Analyze sentiment of a single text"""
        if not self._initialized:
            await self.initialize()
        
        # Check cache
        if use_cache and self.config.enable_caching:
            cache_key = self._get_cache_key(text, language, currency_pairs)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
        
        start_time = time.time()
        
        try:
            # Analyze with ensemble
            result = await self.ensemble_analyzer.analyze_text(text, language, currency_pairs)
            
            # Add trend analysis if enabled
            if self.trend_analyzer and currency_pairs:
                trend_data = await self.trend_analyzer.analyze_trends(
                    result, currency_pairs
                )
                if trend_data:
                    result.market_impact = trend_data.get('market_impact')
                    result.risk_indicators = trend_data.get('risk_indicators', [])
            
            # Cache result
            if use_cache and self.config.enable_caching:
                cache_key = self._get_cache_key(text, language, currency_pairs)
                self._cache_result(cache_key, result)
            
            # Check processing time
            processing_time = (time.time() - start_time) * 1000
            if processing_time > self.config.max_processing_time_ms:
                logger.warning(f"Sentiment analysis took {processing_time:.2f}ms, exceeding limit")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
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
                           currency_pairs_list: Optional[List[List[str]]] = None,
                           use_cache: bool = True) -> SentimentBatchResult:
        """Analyze sentiment of multiple texts"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        results = []
        successful = 0
        failed = 0
        
        try:
            # Process in batches
            batch_size = self.config.batch_size
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_languages = (languages or ["en"] * len(texts))[i:i + batch_size]
                batch_currency_pairs = (currency_pairs_list or [[]] * len(texts))[i:i + batch_size]
                
                # Process batch
                batch_results = await self.ensemble_analyzer.analyze_batch(
                    batch_texts, batch_languages, batch_currency_pairs
                )
                
                # Add trend analysis if enabled
                if self.trend_analyzer:
                    for j, result in enumerate(batch_results):
                        currency_pairs = batch_currency_pairs[j]
                        if currency_pairs:
                            trend_data = await self.trend_analyzer.analyze_trends(
                                result, currency_pairs
                            )
                            if trend_data:
                                result.market_impact = trend_data.get('market_impact')
                                result.risk_indicators = trend_data.get('risk_indicators', [])
                
                results.extend(batch_results)
                successful += len(batch_results)
            
            # Calculate statistics
            total_time = (time.time() - start_time) * 1000
            failed = len(texts) - successful
            avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0
            
            # Create batch result
            batch_result = SentimentBatchResult(
                results=results,
                total_processed=len(texts),
                successful=successful,
                failed=failed,
                processing_time_ms=total_time,
                average_confidence=avg_confidence
            )
            
            logger.info(f"Processed {len(texts)} texts: {successful} successful, {failed} failed in {total_time:.2f}ms")
            return batch_result
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            # Return partial results
            return SentimentBatchResult(
                results=results,
                total_processed=len(texts),
                successful=successful,
                failed=len(texts) - successful,
                processing_time_ms=(time.time() - start_time) * 1000,
                average_confidence=0.0
            )
    
    async def get_sentiment_trends(self, 
                                  currency_pairs: List[str],
                                  timeframe: str = "1h",
                                  hours_back: int = 24) -> List[SentimentTrend]:
        """Get sentiment trends for currency pairs"""
        if not self._initialized:
            await self.initialize()
        
        if not self.trend_analyzer:
            logger.warning("Trend analysis not enabled")
            return []
        
        try:
            trends = await self.trend_analyzer.get_trends(
                currency_pairs, timeframe, hours_back
            )
            return trends
        except Exception as e:
            logger.error(f"Error getting sentiment trends: {e}")
            return []
    
    def _get_cache_key(self, text: str, language: str, currency_pairs: Optional[List[str]]) -> str:
        """Generate cache key for text"""
        key_data = {
            'text': text,
            'language': language,
            'currency_pairs': sorted(currency_pairs or [])
        }
        return json.dumps(key_data, sort_keys=True)
    
    def _get_cached_result(self, cache_key: str) -> Optional[SentimentResult]:
        """Get cached result if valid"""
        if cache_key not in self.cache:
            return None
        
        # Check TTL
        if cache_key in self.cache_timestamps:
            age = datetime.utcnow() - self.cache_timestamps[cache_key]
            if age.total_seconds() > self.config.cache_ttl_seconds:
                # Expired, remove from cache
                del self.cache[cache_key]
                del self.cache_timestamps[cache_key]
                return None
        
        return self.cache[cache_key]
    
    def _cache_result(self, cache_key: str, result: SentimentResult):
        """Cache result"""
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = datetime.utcnow()
        
        # Cleanup old cache entries
        self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Remove expired cache entries"""
        now = datetime.utcnow()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if (now - timestamp).total_seconds() > self.config.cache_ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
    
    async def get_processor_info(self) -> Dict[str, Any]:
        """Get information about the processor"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        info = {
            "status": "initialized",
            "batch_size": self.config.batch_size,
            "max_processing_time_ms": self.config.max_processing_time_ms,
            "enable_trend_analysis": self.config.enable_trend_analysis,
            "enable_caching": self.config.enable_caching,
            "cache_size": len(self.cache),
            "cache_ttl_seconds": self.config.cache_ttl_seconds
        }
        
        # Add analyzer info
        try:
            ensemble_info = await self.ensemble_analyzer.get_ensemble_info()
            info["ensemble_analyzer"] = ensemble_info
        except Exception as e:
            info["ensemble_analyzer"] = {"error": str(e)}
        
        if self.trend_analyzer:
            try:
                trend_info = await self.trend_analyzer.get_trend_info()
                info["trend_analyzer"] = trend_info
            except Exception as e:
                info["trend_analyzer"] = {"error": str(e)}
        
        return info
    
    async def close(self):
        """Close the processor and cleanup resources"""
        try:
            await self.ensemble_analyzer.close()
            if self.trend_analyzer:
                await self.trend_analyzer.close()
        except Exception as e:
            logger.warning(f"Error closing sentiment processor: {e}")
        
        self.cache.clear()
        self.cache_timestamps.clear()
        self._initialized = False
        logger.info("Sentiment processor closed")

