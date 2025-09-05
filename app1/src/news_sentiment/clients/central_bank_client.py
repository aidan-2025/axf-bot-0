#!/usr/bin/env python3
"""
Central Bank Client
Client for central bank RSS feeds and official statements
"""

import asyncio
import logging
import feedparser
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import httpx
import re

from .base_client import BaseNewsClient, RateLimiter
from ..models import NewsSource, NewsArticle, Currency

logger = logging.getLogger(__name__)

class CentralBankClient(BaseNewsClient):
    """Client for central bank RSS feeds and official statements"""
    
    def __init__(self, 
                 rate_limiter: Optional[RateLimiter] = None):
        """Initialize Central Bank client"""
        super().__init__(
            rate_limiter=rate_limiter or RateLimiter(max_requests_per_minute=30, max_requests_per_hour=500)
        )
        
        # Central bank RSS feeds
        self.feeds = {
            'FED': {
                'url': 'https://www.federalreserve.gov/feeds/press_all.xml',
                'currency': Currency.USD,
                'country': 'United States'
            },
            'ECB': {
                'url': 'https://www.ecb.europa.eu/rss/press.html',
                'currency': Currency.EUR,
                'country': 'European Union'
            },
            'BOE': {
                'url': 'https://www.bankofengland.co.uk/news/rss',
                'currency': Currency.GBP,
                'country': 'United Kingdom'
            },
            'BOJ': {
                'url': 'https://www.boj.or.jp/en/news/announcements/index.rss',
                'currency': Currency.JPY,
                'country': 'Japan'
            },
            'SNB': {
                'url': 'https://www.snb.ch/en/mmr/rss',
                'currency': Currency.CHF,
                'country': 'Switzerland'
            },
            'BOC': {
                'url': 'https://www.bankofcanada.ca/rss/',
                'currency': Currency.CAD,
                'country': 'Canada'
            },
            'RBA': {
                'url': 'https://www.rba.gov.au/rss/',
                'currency': Currency.AUD,
                'country': 'Australia'
            },
            'RBNZ': {
                'url': 'https://www.rbnz.govt.nz/news/rss',
                'currency': Currency.NZD,
                'country': 'New Zealand'
            }
        }
    
    @property
    def source(self) -> NewsSource:
        return NewsSource.CENTRAL_BANK
    
    @property
    def name(self) -> str:
        return "Central Bank RSS Feeds"
    
    async def fetch_news(self, 
                        limit: int = 100,
                        since: Optional[datetime] = None,
                        until: Optional[datetime] = None,
                        keywords: Optional[List[str]] = None) -> List[NewsArticle]:
        """Fetch news articles from central bank feeds"""
        articles = []
        
        for bank_name, feed_info in self.feeds.items():
            try:
                bank_articles = await self._fetch_feed_articles(
                    bank_name=bank_name,
                    feed_info=feed_info,
                    limit=limit // len(self.feeds),
                    since=since,
                    until=until,
                    keywords=keywords
                )
                articles.extend(bank_articles)
                
            except Exception as e:
                logger.error(f"Error fetching {bank_name} feed: {e}")
                continue
        
        # Sort by publication date
        articles.sort(key=lambda x: x.published_at, reverse=True)
        
        # Apply limit
        articles = articles[:limit]
        
        logger.info(f"Fetched {len(articles)} central bank articles")
        return articles
    
    async def fetch_events(self,
                          limit: int = 100,
                          since: Optional[datetime] = None,
                          until: Optional[datetime] = None,
                          currency: Optional[str] = None) -> List[NewsArticle]:
        """Central banks don't have traditional events, return news instead"""
        return await self.fetch_news(limit=limit, since=since, until=until)
    
    async def _fetch_feed_articles(self,
                                  bank_name: str,
                                  feed_info: Dict[str, Any],
                                  limit: int = 50,
                                  since: Optional[datetime] = None,
                                  until: Optional[datetime] = None,
                                  keywords: Optional[List[str]] = None) -> List[NewsArticle]:
        """Fetch articles from a specific central bank feed"""
        try:
            # Parse RSS feed
            feed = feedparser.parse(feed_info['url'])
            
            if not feed.entries:
                logger.warning(f"No entries found in {bank_name} feed")
                return []
            
            articles = []
            for entry in feed.entries[:limit]:
                try:
                    article = self._parse_feed_entry(
                        entry=entry,
                        bank_name=bank_name,
                        feed_info=feed_info,
                        since=since,
                        until=until,
                        keywords=keywords
                    )
                    
                    if article:
                        articles.append(article)
                        
                except Exception as e:
                    logger.error(f"Error parsing {bank_name} entry: {e}")
                    continue
            
            logger.info(f"Fetched {len(articles)} articles from {bank_name}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching {bank_name} feed: {e}")
            return []
    
    def _parse_feed_entry(self,
                         entry: Any,
                         bank_name: str,
                         feed_info: Dict[str, Any],
                         since: Optional[datetime] = None,
                         until: Optional[datetime] = None,
                         keywords: Optional[List[str]] = None) -> Optional[NewsArticle]:
        """Parse RSS feed entry into NewsArticle"""
        try:
            # Extract basic information
            title = entry.get('title', '')
            content = entry.get('summary', '') or entry.get('description', '')
            url = entry.get('link', '')
            
            # Parse publication date
            published_at = self._parse_pub_date(entry.get('published', ''))
            if not published_at:
                published_at = datetime.utcnow()
            
            # Filter by date range
            if since and published_at < since:
                return None
            if until and published_at > until:
                return None
            
            # Extract content
            if hasattr(entry, 'content') and entry.content:
                content = entry.content[0].value if isinstance(entry.content, list) else str(entry.content)
            
            # Clean content
            content = self._clean_html_content(content)
            
            # Check keywords if provided
            if keywords:
                text_to_search = f"{title} {content}".lower()
                if not any(keyword.lower() in text_to_search for keyword in keywords):
                    return None
            
            # Determine relevance to forex
            relevance_score = self._calculate_relevance_score(title, content)
            
            # Extract keywords
            article_keywords = self._extract_keywords(title, content)
            
            # Determine currency pairs
            currency_pairs = self._get_currency_pairs(feed_info['currency'])
            
            # Generate article ID
            article_id = f"cb_{bank_name}_{hashlib.md5(f'{title}{published_at}'.encode()).hexdigest()[:12]}"
            
            return NewsArticle(
                article_id=article_id,
                title=title,
                content=content,
                summary=self._generate_summary(content),
                source=self.source,
                author=f"{bank_name}",
                published_at=published_at,
                language='en',
                url=url,
                currency_pairs=currency_pairs,
                keywords=article_keywords,
                relevance_score=relevance_score,
                raw_data={
                    'bank': bank_name,
                    'currency': feed_info['currency'].value,
                    'country': feed_info['country'],
                    'feed_url': feed_info['url']
                }
            )
            
        except Exception as e:
            logger.error(f"Error parsing feed entry: {e}")
            return None
    
    def _parse_pub_date(self, pub_date_str: str) -> Optional[datetime]:
        """Parse publication date from various formats"""
        if not pub_date_str:
            return None
        
        try:
            # Try parsing with feedparser's date parsing
            import time
            parsed_time = time.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S %Z')
            return datetime(*parsed_time[:6])
        except ValueError:
            try:
                # Try ISO format
                return datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
            except ValueError:
                try:
                    # Try other common formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M:%S']:
                        try:
                            return datetime.strptime(pub_date_str, fmt)
                        except ValueError:
                            continue
                except:
                    pass
        
        return None
    
    def _clean_html_content(self, content: str) -> str:
        """Clean HTML content and extract text"""
        if not content:
            return ""
        
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', content)
        
        # Decode HTML entities
        import html
        clean = html.unescape(clean)
        
        # Clean up whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        return clean
    
    def _generate_summary(self, content: str, max_length: int = 200) -> str:
        """Generate summary from content"""
        if not content:
            return ""
        
        # Simple summary: first sentence or first N characters
        sentences = content.split('. ')
        if sentences:
            summary = sentences[0]
            if len(summary) > max_length:
                summary = summary[:max_length-3] + '...'
            return summary
        
        # Fallback to truncation
        if len(content) > max_length:
            return content[:max_length-3] + '...'
        
        return content
    
    def _calculate_relevance_score(self, title: str, content: str) -> float:
        """Calculate relevance score for forex trading"""
        text = f"{title} {content}".lower()
        
        # Forex-related keywords and their weights
        forex_keywords = {
            'interest rate': 0.3,
            'monetary policy': 0.3,
            'inflation': 0.25,
            'gdp': 0.2,
            'unemployment': 0.2,
            'trade balance': 0.15,
            'current account': 0.15,
            'fiscal policy': 0.2,
            'quantitative easing': 0.25,
            'taper': 0.2,
            'dovish': 0.15,
            'hawkish': 0.15,
            'currency': 0.1,
            'exchange rate': 0.1,
            'forex': 0.1,
            'fx': 0.1
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
        
        # Common forex and economic keywords
        keywords = []
        
        # Economic indicators
        indicators = ['gdp', 'inflation', 'unemployment', 'interest rate', 'trade balance', 
                     'current account', 'retail sales', 'manufacturing', 'services', 'consumer confidence']
        
        for indicator in indicators:
            if indicator in text:
                keywords.append(indicator)
        
        # Policy terms
        policy_terms = ['monetary policy', 'fiscal policy', 'quantitative easing', 'taper', 
                       'dovish', 'hawkish', 'accommodative', 'restrictive']
        
        for term in policy_terms:
            if term in text:
                keywords.append(term)
        
        # Currency terms
        currency_terms = ['currency', 'exchange rate', 'forex', 'fx', 'dollar', 'euro', 'pound', 'yen']
        
        for term in currency_terms:
            if term in text:
                keywords.append(term)
        
        return list(set(keywords))  # Remove duplicates
    
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
    
    async def get_high_impact_news(self, 
                                 hours_ahead: int = 24) -> List[NewsArticle]:
        """Get high impact central bank news"""
        since = datetime.utcnow() - timedelta(hours=1)
        until = datetime.utcnow() + timedelta(hours=hours_ahead)
        
        articles = await self.fetch_news(since=since, until=until)
        
        # Filter for high relevance articles
        high_impact_articles = [
            article for article in articles 
            if article.relevance_score > 0.3
        ]
        
        logger.info(f"Found {len(high_impact_articles)} high impact central bank articles")
        return high_impact_articles
