#!/usr/bin/env python3
"""
News Ingestion Service
Orchestrates news and sentiment data collection from multiple sources
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import json

from .models import (
    NewsArticle, EconomicEvent, NewsIngestionConfig, NewsIngestionStats,
    NewsSource, SentimentLabel
)
from .clients import (
    ReutersClient, BloombergClient, ForexFactoryClient, CentralBankClient,
    TwitterClient, FinageClient, AlphaVantageClient, FinnhubClient
)
from .sentiment_analyzer import SentimentAnalyzer
from .news_processor import NewsProcessor

logger = logging.getLogger(__name__)

class NewsIngestionService:
    """Service for ingesting news and sentiment data from multiple sources"""
    
    def __init__(self, config: NewsIngestionConfig):
        """Initialize news ingestion service"""
        self.config = config
        self.stats = NewsIngestionStats()
        
        # Initialize clients
        self.clients = self._initialize_clients()
        
        # Initialize processors
        self.sentiment_analyzer = SentimentAnalyzer() if config.enable_sentiment_analysis else None
        self.news_processor = NewsProcessor(config)
        
        # Data storage
        self.articles_buffer: List[NewsArticle] = []
        self.events_buffer: List[EconomicEvent] = []
        self.processed_hashes: Set[str] = set()
        
        # Background tasks
        self._ingestion_task: Optional[asyncio.Task] = None
        self._processing_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    def _initialize_clients(self) -> Dict[NewsSource, Any]:
        """Initialize news API clients based on configuration"""
        clients = {}
        
        # Initialize available clients
        if self.config.reuters_api_key:
            clients[NewsSource.REUTERS] = ReutersClient(self.config.reuters_api_key)
        
        if self.config.bloomberg_api_key:
            clients[NewsSource.BLOOMBERG] = BloombergClient(self.config.bloomberg_api_key)
        
        # Forex Factory doesn't require API key
        clients[NewsSource.FOREX_FACTORY] = ForexFactoryClient()
        
        # Central Bank feeds don't require API key
        clients[NewsSource.CENTRAL_BANK] = CentralBankClient()
        
        if self.config.twitter_bearer_token:
            clients[NewsSource.TWITTER] = TwitterClient(self.config.twitter_bearer_token)
        
        if self.config.finage_api_key:
            clients[NewsSource.FINAGE] = FinageClient(self.config.finage_api_key)
        
        if self.config.alpha_vantage_api_key:
            clients[NewsSource.ALPHA_VANTAGE] = AlphaVantageClient(self.config.alpha_vantage_api_key)
        
        if self.config.finnhub_api_key:
            clients[NewsSource.FINNHUB] = FinnhubClient(self.config.finnhub_api_key)
        
        logger.info(f"Initialized {len(clients)} news clients")
        return clients
    
    async def start(self):
        """Start the news ingestion service"""
        if self._is_running:
            logger.warning("News ingestion service is already running")
            return
        
        self._is_running = True
        logger.info("Starting news ingestion service")
        
        # Start background tasks
        self._ingestion_task = asyncio.create_task(self._ingestion_loop())
        self._processing_task = asyncio.create_task(self._processing_loop())
        
        logger.info("News ingestion service started")
    
    async def stop(self):
        """Stop the news ingestion service"""
        if not self._is_running:
            logger.warning("News ingestion service is not running")
            return
        
        self._is_running = False
        logger.info("Stopping news ingestion service")
        
        # Cancel background tasks
        if self._ingestion_task:
            self._ingestion_task.cancel()
            try:
                await self._ingestion_task
            except asyncio.CancelledError:
                pass
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        # Close all clients
        for client in self.clients.values():
            await client.close()
        
        logger.info("News ingestion service stopped")
    
    async def _ingestion_loop(self):
        """Main ingestion loop"""
        while self._is_running:
            try:
                # Fetch news from all sources
                await self._fetch_all_news()
                
                # Fetch events from all sources
                await self._fetch_all_events()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Fetch every minute
                
            except Exception as e:
                logger.error(f"Error in ingestion loop: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def _processing_loop(self):
        """Main processing loop"""
        while self._is_running:
            try:
                # Process buffered articles
                if self.articles_buffer:
                    await self._process_articles()
                
                # Process buffered events
                if self.events_buffer:
                    await self._process_events()
                
                # Wait before next iteration
                await asyncio.sleep(self.config.flush_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def _fetch_all_news(self):
        """Fetch news from all available sources"""
        tasks = []
        
        for source, client in self.clients.items():
            if hasattr(client, 'fetch_news'):
                task = asyncio.create_task(
                    self._fetch_news_from_source(source, client)
                )
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _fetch_all_events(self):
        """Fetch events from all available sources"""
        tasks = []
        
        for source, client in self.clients.items():
            if hasattr(client, 'fetch_events'):
                task = asyncio.create_task(
                    self._fetch_events_from_source(source, client)
                )
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _fetch_news_from_source(self, source: NewsSource, client: Any):
        """Fetch news from a specific source"""
        try:
            # Calculate time range
            since = datetime.utcnow() - timedelta(hours=1)
            until = datetime.utcnow()
            
            # Fetch news
            articles = await client.fetch_news(
                limit=100,
                since=since,
                until=until,
                keywords=self.config.forex_keywords
            )
            
            # Filter by relevance
            relevant_articles = [
                article for article in articles
                if article.relevance_score >= self.config.min_relevance_score
            ]
            
            # Add to buffer
            self.articles_buffer.extend(relevant_articles)
            
            # Update statistics
            self.stats.total_articles += len(articles)
            self.stats.articles_by_source[source.value] = self.stats.articles_by_source.get(source.value, 0) + len(articles)
            self.stats.last_ingestion = datetime.utcnow()
            
            logger.info(f"Fetched {len(relevant_articles)} relevant articles from {source.value}")
            
        except Exception as e:
            logger.error(f"Error fetching news from {source.value}: {e}")
            self.stats.errors_by_source[source.value] = self.stats.errors_by_source.get(source.value, 0) + 1
    
    async def _fetch_events_from_source(self, source: NewsSource, client: Any):
        """Fetch events from a specific source"""
        try:
            # Calculate time range
            since = datetime.utcnow() - timedelta(days=1)
            until = datetime.utcnow() + timedelta(days=7)
            
            # Fetch events
            events = await client.fetch_events(
                limit=100,
                since=since,
                until=until
            )
            
            # Add to buffer
            self.events_buffer.extend(events)
            
            logger.info(f"Fetched {len(events)} events from {source.value}")
            
        except Exception as e:
            logger.error(f"Error fetching events from {source.value}: {e}")
            self.stats.errors_by_source[source.value] = self.stats.errors_by_source.get(source.value, 0) + 1
    
    async def _process_articles(self):
        """Process buffered articles"""
        if not self.articles_buffer:
            return
        
        # Process in batches
        batch_size = min(self.config.batch_size, len(self.articles_buffer))
        batch = self.articles_buffer[:batch_size]
        self.articles_buffer = self.articles_buffer[batch_size:]
        
        processed_count = 0
        failed_count = 0
        duplicate_count = 0
        
        for article in batch:
            try:
                # Check for duplicates
                if self.config.enable_deduplication and article.content_hash in self.processed_hashes:
                    duplicate_count += 1
                    continue
                
                # Process article
                processed_article = await self.news_processor.process_article(article)
                
                if processed_article:
                    # Perform sentiment analysis if enabled
                    if self.config.enable_sentiment_analysis and self.sentiment_analyzer:
                        sentiment_data = await self.sentiment_analyzer.analyze_text(
                            text=processed_article.content,
                            language=processed_article.language
                        )
                        
                        if sentiment_data:
                            processed_article.sentiment_label = sentiment_data.label
                            processed_article.sentiment_score = sentiment_data.score
                            processed_article.confidence = sentiment_data.confidence
                    
                    # Store processed article (placeholder - implement actual storage)
                    await self._store_article(processed_article)
                    
                    # Mark as processed
                    if self.config.enable_deduplication:
                        self.processed_hashes.add(processed_article.content_hash)
                    
                    processed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing article {article.article_id}: {e}")
                failed_count += 1
        
        # Update statistics
        self.stats.processed_articles += processed_count
        self.stats.failed_articles += failed_count
        self.stats.duplicate_articles += duplicate_count
        
        logger.info(f"Processed {processed_count} articles, {failed_count} failed, {duplicate_count} duplicates")
    
    async def _process_events(self):
        """Process buffered events"""
        if not self.events_buffer:
            return
        
        # Process in batches
        batch_size = min(self.config.batch_size, len(self.events_buffer))
        batch = self.events_buffer[:batch_size]
        self.events_buffer = self.events_buffer[batch_size:]
        
        processed_count = 0
        failed_count = 0
        
        for event in batch:
            try:
                # Process event
                processed_event = await self.news_processor.process_event(event)
                
                if processed_event:
                    # Store processed event (placeholder - implement actual storage)
                    await self._store_event(processed_event)
                    processed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing event {event.event_id}: {e}")
                failed_count += 1
        
        logger.info(f"Processed {processed_count} events, {failed_count} failed")
    
    async def _store_article(self, article: NewsArticle):
        """Store processed article (placeholder)"""
        # TODO: Implement actual storage to database
        logger.debug(f"Storing article: {article.article_id}")
        pass
    
    async def _store_event(self, event: EconomicEvent):
        """Store processed event (placeholder)"""
        # TODO: Implement actual storage to database
        logger.debug(f"Storing event: {event.event_id}")
        pass
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all clients"""
        health_status = {
            'service_running': self._is_running,
            'clients': {},
            'statistics': self.stats.to_dict()
        }
        
        # Check each client
        for source, client in self.clients.items():
            try:
                client_health = await client.health_check()
                health_status['clients'][source.value] = client_health
            except Exception as e:
                health_status['clients'][source.value] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return health_status
    
    async def get_recent_news(self, 
                            limit: int = 100,
                            source: Optional[NewsSource] = None,
                            min_relevance: float = 0.0) -> List[NewsArticle]:
        """Get recent news articles"""
        # TODO: Implement actual retrieval from database
        logger.info(f"Getting recent news: limit={limit}, source={source}, min_relevance={min_relevance}")
        return []
    
    async def get_upcoming_events(self, 
                                hours_ahead: int = 24,
                                currency: Optional[str] = None) -> List[EconomicEvent]:
        """Get upcoming economic events"""
        logger.info(f"Getting upcoming events: hours_ahead={hours_ahead}, currency={currency}")
        
        # Filter events by time and currency
        now = datetime.utcnow()
        cutoff_time = now + timedelta(hours=hours_ahead)
        
        filtered_events = []
        for event in self.events_buffer:
            # Check time filter
            if event.event_time > now and event.event_time <= cutoff_time:
                # Check currency filter
                if currency is None or event.currency.value.upper() == currency.upper():
                    filtered_events.append(event)
        
        # Sort by event time
        filtered_events.sort(key=lambda x: x.event_time)
        
        logger.info(f"Found {len(filtered_events)} upcoming events")
        return filtered_events
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        return self.stats.to_dict()
    
    async def force_refresh(self):
        """Force refresh of all data sources"""
        logger.info("Forcing refresh of all data sources")
        await self._fetch_all_news()
        await self._fetch_all_events()
        await self._process_articles()
        await self._process_events()
