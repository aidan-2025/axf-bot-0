#!/usr/bin/env python3
"""
Alpha Vantage Client
Client for Alpha Vantage news and market data API
"""

import asyncio
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import httpx

from .base_client import BaseNewsClient, RateLimiter
from ..models import NewsSource, NewsArticle, EconomicEvent, ImpactLevel, Currency

logger = logging.getLogger(__name__)

class AlphaVantageClient(BaseNewsClient):
    """Client for Alpha Vantage API"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://www.alphavantage.co/query",
                 rate_limiter: Optional[RateLimiter] = None):
        """Initialize Alpha Vantage client"""
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            rate_limiter=rate_limiter or RateLimiter(max_requests_per_minute=5, max_requests_per_hour=500)
        )
    
    @property
    def source(self) -> NewsSource:
        return NewsSource.ALPHA_VANTAGE
    
    @property
    def name(self) -> str:
        return "Alpha Vantage API"
    
    async def fetch_news(self, 
                        limit: int = 100,
                        since: Optional[datetime] = None,
                        until: Optional[datetime] = None,
                        keywords: Optional[List[str]] = None) -> List[NewsArticle]:
        """Fetch news articles from Alpha Vantage"""
        try:
            # Alpha Vantage news endpoint
            params = {
                'function': 'NEWS_SENTIMENT',
                'apikey': self.api_key,
                'limit': min(limit, 1000),
                'sort': 'LATEST'
            }
            
            if keywords:
                # Alpha Vantage supports topic filtering
                topics = ','.join(keywords[:5])  # Limit to 5 topics
                params['topics'] = topics
            
            # Make request
            response = await self._make_request('GET', self.base_url, params=params)
            
            # Parse response
            data = response.json()
            articles = []
            
            if 'feed' in data:
                for article_data in data['feed'][:limit]:
                    article = self._parse_article(article_data)
                    if article:
                        articles.append(article)
            
            logger.info(f"Fetched {len(articles)} news articles from Alpha Vantage")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return []
    
    async def fetch_events(self,
                          limit: int = 100,
                          since: Optional[datetime] = None,
                          until: Optional[datetime] = None,
                          currency: Optional[str] = None) -> List[EconomicEvent]:
        """Alpha Vantage doesn't have economic events, return empty list"""
        logger.info("Alpha Vantage doesn't provide economic events")
        return []
    
    def _parse_article(self, article_data: Dict[str, Any]) -> Optional[NewsArticle]:
        """Parse article data from API response"""
        try:
            # Extract basic information
            article_id = str(article_data.get('uuid', ''))
            title = article_data.get('title', '')
            content = article_data.get('summary', '')
            url = article_data.get('url', '')
            
            # Parse publication date
            published_at_str = article_data.get('time_published', '')
            try:
                # Alpha Vantage format: 20240101T120000
                published_at = datetime.strptime(published_at_str, '%Y%m%dT%H%M%S')
            except ValueError:
                published_at = datetime.utcnow()
            
            # Extract author and source
            author = article_data.get('authors', ['Alpha Vantage'])[0] if article_data.get('authors') else 'Alpha Vantage'
            source_name = article_data.get('source', 'Alpha Vantage')
            
            # Extract sentiment data
            sentiment_data = article_data.get('overall_sentiment_score', 0.0)
            sentiment_label = article_data.get('overall_sentiment_label', 'neutral')
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(title, content)
            
            # Extract keywords and currency pairs
            keywords = self._extract_keywords(title, content)
            currency_pairs = self._extract_currency_pairs(title, content)
            
            # Generate article ID if not provided
            if not article_id:
                content_hash = hashlib.md5(f'{title}{published_at}'.encode()).hexdigest()[:12]
                article_id = f"av_{content_hash}"
            
            return NewsArticle(
                article_id=article_id,
                title=title,
                content=content,
                summary=self._generate_summary(content),
                source=self.source,
                author=author,
                published_at=published_at,
                language='en',
                url=url,
                currency_pairs=currency_pairs,
                keywords=keywords,
                relevance_score=relevance_score,
                sentiment_score=sentiment_data,
                raw_data=article_data
            )
            
        except Exception as e:
            logger.error(f"Error parsing Alpha Vantage article: {e}")
            return None
    
    def _calculate_relevance_score(self, title: str, content: str) -> float:
        """Calculate relevance score for forex trading"""
        text = f"{title} {content}".lower()
        
        # Forex-related keywords and their weights
        forex_keywords = {
            'forex': 0.2, 'fx': 0.2, 'currency': 0.15, 'exchange rate': 0.15,
            'central bank': 0.25, 'interest rate': 0.2, 'inflation': 0.15,
            'gdp': 0.1, 'unemployment': 0.1, 'trade balance': 0.1
        }
        
        score = 0.0
        for keyword, weight in forex_keywords.items():
            if keyword in text:
                score += weight
        
        # Normalize to 0-1 range
        return min(score, 1.0)
    
    def _extract_keywords(self, title: str, content: str) -> List[str]:
        """Extract relevant keywords from title and content"""
        text = f"{title} {content}".lower()
        keywords = []
        
        # Economic indicators
        indicators = ['gdp', 'inflation', 'unemployment', 'interest rate', 'trade balance']
        
        for indicator in indicators:
            if indicator in text:
                keywords.append(indicator)
        
        return list(set(keywords))
    
    def _extract_currency_pairs(self, title: str, content: str) -> List[str]:
        """Extract currency pairs from title and content"""
        text_upper = f"{title} {content}".upper()
        currency_pairs = []
        
        # Major pairs
        major_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD'
        ]
        
        for pair in major_pairs:
            if pair in text_upper:
                currency_pairs.append(pair)
        
        return list(set(currency_pairs))
    
    def _generate_summary(self, content: str, max_length: int = 200) -> str:
        """Generate summary from content"""
        if not content:
            return ""
        
        if len(content) > max_length:
            return content[:max_length-3] + '...'
        
        return content

