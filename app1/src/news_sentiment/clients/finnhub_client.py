#!/usr/bin/env python3
"""
Finnhub Client
Client for Finnhub news and sentiment API
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

class FinnhubClient(BaseNewsClient):
    """Client for Finnhub API"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://finnhub.io/api/v1",
                 rate_limiter: Optional[RateLimiter] = None):
        """Initialize Finnhub client"""
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            rate_limiter=rate_limiter or RateLimiter(max_requests_per_minute=60, max_requests_per_hour=1000)
        )
    
    @property
    def source(self) -> NewsSource:
        return NewsSource.FINNHUB
    
    @property
    def name(self) -> str:
        return "Finnhub API"
    
    async def fetch_news(self, 
                        limit: int = 100,
                        since: Optional[datetime] = None,
                        until: Optional[datetime] = None,
                        keywords: Optional[List[str]] = None) -> List[NewsArticle]:
        """Fetch news articles from Finnhub"""
        try:
            # Prepare parameters
            params = {
                'token': self.api_key,
                'category': 'general'
            }
            
            if since:
                params['from'] = int(since.timestamp())
            if until:
                params['to'] = int(until.timestamp())
            
            # Make request
            url = f"{self.base_url}/news"
            response = await self._make_request('GET', url, params=params)
            
            # Parse response
            data = response.json()
            articles = []
            
            if isinstance(data, list):
                for article_data in data[:limit]:
                    article = self._parse_article(article_data)
                    if article:
                        articles.append(article)
            
            logger.info(f"Fetched {len(articles)} news articles from Finnhub")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub news: {e}")
            return []
    
    async def fetch_events(self,
                          limit: int = 100,
                          since: Optional[datetime] = None,
                          until: Optional[datetime] = None,
                          currency: Optional[str] = None) -> List[EconomicEvent]:
        """Fetch economic events from Finnhub"""
        try:
            # Prepare parameters
            params = {
                'token': self.api_key
            }
            
            if since:
                params['from'] = since.strftime('%Y-%m-%d')
            if until:
                params['to'] = until.strftime('%Y-%m-%d')
            
            # Make request
            url = f"{self.base_url}/calendar/economic"
            response = await self._make_request('GET', url, params=params)
            
            # Parse response
            data = response.json()
            events = []
            
            if 'economicCalendar' in data:
                for event_data in data['economicCalendar'][:limit]:
                    event = self._parse_event(event_data)
                    if event:
                        events.append(event)
            
            logger.info(f"Fetched {len(events)} economic events from Finnhub")
            return events
            
        except Exception as e:
            logger.error(f"Error fetching Finnhub events: {e}")
            return []
    
    def _parse_article(self, article_data: Dict[str, Any]) -> Optional[NewsArticle]:
        """Parse article data from API response"""
        try:
            # Extract basic information
            article_id = str(article_data.get('id', ''))
            title = article_data.get('headline', '')
            content = article_data.get('summary', '')
            url = article_data.get('url', '')
            
            # Parse publication date
            published_at_str = article_data.get('datetime', '')
            try:
                published_at = datetime.fromtimestamp(int(published_at_str) / 1000)
            except (ValueError, TypeError):
                published_at = datetime.utcnow()
            
            # Extract author and source
            author = article_data.get('author', 'Finnhub')
            source_name = article_data.get('source', 'Finnhub')
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(title, content)
            
            # Extract keywords and currency pairs
            keywords = self._extract_keywords(title, content)
            currency_pairs = self._extract_currency_pairs(title, content)
            
            # Generate article ID if not provided
            if not article_id:
                content_hash = hashlib.md5(f'{title}{published_at}'.encode()).hexdigest()[:12]
                article_id = f"finnhub_{content_hash}"
            
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
                raw_data=article_data
            )
            
        except Exception as e:
            logger.error(f"Error parsing Finnhub article: {e}")
            return None
    
    def _parse_event(self, event_data: Dict[str, Any]) -> Optional[EconomicEvent]:
        """Parse event data from API response"""
        try:
            # Extract basic information
            event_id = str(event_data.get('id', ''))
            title = event_data.get('title', '')
            description = event_data.get('description', '')
            
            # Parse event time
            event_time_str = event_data.get('datetime', '')
            try:
                event_time = datetime.fromtimestamp(int(event_time_str) / 1000)
            except (ValueError, TypeError):
                event_time = datetime.utcnow()
            
            # Parse impact level
            impact_str = event_data.get('impact', 'low').lower()
            impact_mapping = {
                'low': ImpactLevel.LOW,
                'medium': ImpactLevel.MEDIUM,
                'high': ImpactLevel.HIGH,
                'very_high': ImpactLevel.VERY_HIGH
            }
            impact = impact_mapping.get(impact_str, ImpactLevel.LOW)
            
            # Parse currency
            currency_str = event_data.get('currency', 'USD')
            try:
                currency = Currency(currency_str.upper())
            except ValueError:
                currency = Currency.USD
            
            # Extract data values
            actual = event_data.get('actual')
            forecast = event_data.get('forecast')
            previous = event_data.get('previous')
            unit = event_data.get('unit', '')
            
            # Convert to float if possible
            if actual is not None:
                try:
                    actual = float(actual)
                except (ValueError, TypeError):
                    actual = None
            
            if forecast is not None:
                try:
                    forecast = float(forecast)
                except (ValueError, TypeError):
                    forecast = None
            
            if previous is not None:
                try:
                    previous = float(previous)
                except (ValueError, TypeError):
                    previous = None
            
            # Determine currency pairs
            currency_pairs = self._get_currency_pairs(currency)
            
            # Extract additional metadata
            country = event_data.get('country', '')
            category = event_data.get('category', '')
            
            return EconomicEvent(
                event_id=event_id,
                title=title,
                description=description,
                event_time=event_time,
                timezone='UTC',
                impact=impact,
                currency=currency,
                currency_pairs=currency_pairs,
                actual=actual,
                forecast=forecast,
                previous=previous,
                unit=unit,
                source=self.source,
                country=country,
                category=category
            )
            
        except Exception as e:
            logger.error(f"Error parsing Finnhub event: {e}")
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
    
    def _get_currency_pairs(self, currency: Currency) -> List[str]:
        """Get relevant currency pairs for a currency"""
        major_pairs = {
            Currency.USD: ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD'],
            Currency.EUR: ['EURUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURCAD', 'EURAUD', 'EURNZD'],
            Currency.GBP: ['GBPUSD', 'EURGBP', 'GBPJPY', 'GBPCHF', 'GBPCAD', 'GBPAUD', 'GBPNZD'],
            Currency.JPY: ['USDJPY', 'EURJPY', 'GBPJPY', 'CHFJPY', 'CADJPY', 'AUDJPY', 'NZDJPY'],
            Currency.CHF: ['USDCHF', 'EURCHF', 'GBPCHF', 'CHFJPY', 'CHFCAD', 'CHFAUD', 'CHFNZD'],
            Currency.CAD: ['USDCAD', 'EURCAD', 'GBPCAD', 'CADJPY', 'CHFCAD', 'AUDCAD', 'NZDCAD'],
            Currency.AUD: ['AUDUSD', 'EURAUD', 'GBPAUD', 'AUDJPY', 'CHFAUD', 'AUDCAD', 'AUDNZD'],
            Currency.NZD: ['NZDUSD', 'EURNZD', 'GBPNZD', 'NZDJPY', 'CHFNZD', 'NZDCAD', 'AUDNZD']
        }
        
        return major_pairs.get(currency, [f'{currency.value}USD'])
    
    def _generate_summary(self, content: str, max_length: int = 200) -> str:
        """Generate summary from content"""
        if not content:
            return ""
        
        if len(content) > max_length:
            return content[:max_length-3] + '...'
        
        return content

