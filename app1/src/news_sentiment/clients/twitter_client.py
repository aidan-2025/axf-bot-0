#!/usr/bin/env python3
"""
Twitter Client
Client for Twitter/X API v2 for financial sentiment analysis
"""

import asyncio
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import httpx
import re

from .base_client import BaseNewsClient, RateLimiter
from ..models import NewsSource, NewsArticle, Currency

logger = logging.getLogger(__name__)

class TwitterClient(BaseNewsClient):
    """Client for Twitter/X API v2"""
    
    def __init__(self, 
                 bearer_token: str,
                 base_url: str = "https://api.twitter.com/2",
                 rate_limiter: Optional[RateLimiter] = None):
        """Initialize Twitter client"""
        super().__init__(
            api_key=bearer_token,
            base_url=base_url,
            rate_limiter=rate_limiter or RateLimiter(max_requests_per_minute=50, max_requests_per_hour=1000)
        )
        
        # Forex-related search terms
        self.forex_keywords = [
            'forex', 'fx', 'currency', 'exchange rate', 'central bank',
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD',
            'EURGBP', 'EURJPY', 'GBPJPY', 'CHFJPY', 'CADJPY', 'AUDJPY',
            'interest rate', 'inflation', 'gdp', 'unemployment', 'trade balance',
            'fed', 'ecb', 'boe', 'boj', 'snb', 'boc', 'rba', 'rbnz'
        ]
        
        # Language codes for supported languages
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja']
    
    @property
    def source(self) -> NewsSource:
        return NewsSource.TWITTER
    
    @property
    def name(self) -> str:
        return "Twitter/X API v2"
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for Twitter API"""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': 'ForexBot/1.0',
            'Accept': 'application/json'
        }
    
    async def fetch_news(self, 
                        limit: int = 100,
                        since: Optional[datetime] = None,
                        until: Optional[datetime] = None,
                        keywords: Optional[List[str]] = None) -> List[NewsArticle]:
        """Fetch tweets as news articles"""
        try:
            # Build search query
            query = self._build_search_query(keywords)
            
            # Prepare parameters
            params = {
                'query': query,
                'max_results': min(limit, 100),  # API limit per request
                'tweet.fields': 'created_at,author_id,public_metrics,lang,context_annotations',
                'user.fields': 'username,verified,public_metrics',
                'expansions': 'author_id'
            }
            
            if since:
                params['start_time'] = since.isoformat()
            if until:
                params['end_time'] = until.isoformat()
            
            # Make request
            url = f"{self.base_url}/tweets/search/recent"
            response = await self._make_request('GET', url, params=params)
            
            # Parse response
            data = response.json()
            articles = []
            
            if 'data' in data:
                # Get user information
                users = {}
                if 'includes' in data and 'users' in data['includes']:
                    users = {user['id']: user for user in data['includes']['users']}
                
                for tweet_data in data['data']:
                    article = self._parse_tweet(tweet_data, users)
                    if article:
                        articles.append(article)
            
            logger.info(f"Fetched {len(articles)} tweets from Twitter")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching Twitter data: {e}")
            return []
    
    async def fetch_events(self,
                          limit: int = 100,
                          since: Optional[datetime] = None,
                          until: Optional[datetime] = None,
                          currency: Optional[str] = None) -> List[NewsArticle]:
        """Twitter doesn't have traditional events, return tweets instead"""
        return await self.fetch_news(limit=limit, since=since, until=until)
    
    def _build_search_query(self, keywords: Optional[List[str]] = None) -> str:
        """Build Twitter search query"""
        # Base forex keywords
        base_keywords = self.forex_keywords[:10]  # Limit to avoid query too long
        
        if keywords:
            # Combine with provided keywords
            all_keywords = list(set(base_keywords + keywords))
        else:
            all_keywords = base_keywords
        
        # Build OR query
        query_parts = []
        for keyword in all_keywords[:20]:  # Twitter query limit
            if ' ' in keyword:
                query_parts.append(f'"{keyword}"')
            else:
                query_parts.append(keyword)
        
        query = ' OR '.join(query_parts)
        
        # Add language filter
        lang_filter = ' OR '.join([f'lang:{lang}' for lang in self.supported_languages[:5]])
        query = f'({query}) ({lang_filter})'
        
        # Exclude retweets and replies for better quality
        query += ' -is:retweet -is:reply'
        
        return query
    
    def _parse_tweet(self, tweet_data: Dict[str, Any], users: Dict[str, Any]) -> Optional[NewsArticle]:
        """Parse tweet data into NewsArticle"""
        try:
            # Extract basic information
            tweet_id = tweet_data.get('id', '')
            text = tweet_data.get('text', '')
            created_at = tweet_data.get('created_at', '')
            author_id = tweet_data.get('author_id', '')
            language = tweet_data.get('lang', 'en')
            
            # Parse creation date
            try:
                published_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except ValueError:
                published_at = datetime.utcnow()
            
            # Get user information
            author = users.get(author_id, {})
            username = author.get('username', 'unknown')
            verified = author.get('verified', False)
            followers_count = author.get('public_metrics', {}).get('followers_count', 0)
            
            # Get tweet metrics
            metrics = tweet_data.get('public_metrics', {})
            retweet_count = metrics.get('retweet_count', 0)
            like_count = metrics.get('like_count', 0)
            reply_count = metrics.get('reply_count', 0)
            
            # Calculate relevance score
            relevance_score = self._calculate_tweet_relevance(text, verified, followers_count, metrics)
            
            # Extract currency pairs and keywords
            currency_pairs = self._extract_currency_pairs(text)
            keywords = self._extract_keywords(text)
            
            # Generate article ID
            article_id = f"tw_{tweet_id}"
            
            # Create URL
            url = f"https://twitter.com/{username}/status/{tweet_id}"
            
            # Generate summary (truncated text)
            summary = text[:200] + '...' if len(text) > 200 else text
            
            return NewsArticle(
                article_id=article_id,
                title=f"Tweet by @{username}",
                content=text,
                summary=summary,
                source=self.source,
                author=f"@{username}",
                published_at=published_at,
                language=language,
                url=url,
                currency_pairs=currency_pairs,
                keywords=keywords,
                relevance_score=relevance_score,
                raw_data={
                    'tweet_id': tweet_id,
                    'author_id': author_id,
                    'username': username,
                    'verified': verified,
                    'followers_count': followers_count,
                    'retweet_count': retweet_count,
                    'like_count': like_count,
                    'reply_count': reply_count,
                    'language': language
                }
            )
            
        except Exception as e:
            logger.error(f"Error parsing tweet: {e}")
            return None
    
    def _calculate_tweet_relevance(self, 
                                 text: str, 
                                 verified: bool, 
                                 followers_count: int, 
                                 metrics: Dict[str, Any]) -> float:
        """Calculate relevance score for a tweet"""
        score = 0.0
        text_lower = text.lower()
        
        # Base score from forex keywords
        forex_keywords = {
            'forex': 0.2, 'fx': 0.2, 'currency': 0.15, 'exchange rate': 0.15,
            'central bank': 0.25, 'interest rate': 0.2, 'inflation': 0.15,
            'gdp': 0.1, 'unemployment': 0.1, 'trade balance': 0.1
        }
        
        for keyword, weight in forex_keywords.items():
            if keyword in text_lower:
                score += weight
        
        # Currency pair mentions
        currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD']
        for pair in currency_pairs:
            if pair in text_lower:
                score += 0.1
        
        # Author credibility boost
        if verified:
            score += 0.2
        
        # Follower count boost (logarithmic)
        if followers_count > 10000:
            score += 0.1
        elif followers_count > 1000:
            score += 0.05
        
        # Engagement boost
        total_engagement = (
            metrics.get('retweet_count', 0) + 
            metrics.get('like_count', 0) + 
            metrics.get('reply_count', 0)
        )
        
        if total_engagement > 100:
            score += 0.1
        elif total_engagement > 10:
            score += 0.05
        
        # Normalize to 0-1 range
        return min(score, 1.0)
    
    def _extract_currency_pairs(self, text: str) -> List[str]:
        """Extract currency pairs from tweet text"""
        text_upper = text.upper()
        currency_pairs = []
        
        # Major pairs
        major_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD',
            'EURGBP', 'EURJPY', 'EURCHF', 'EURCAD', 'EURAUD', 'EURNZD',
            'GBPJPY', 'GBPCHF', 'GBPCAD', 'GBPAUD', 'GBPNZD',
            'CHFJPY', 'CADJPY', 'AUDJPY', 'NZDJPY',
            'CHFCAD', 'CHFAUD', 'CHFNZD',
            'AUDCAD', 'AUDNZD', 'NZDCAD'
        ]
        
        for pair in major_pairs:
            if pair in text_upper:
                currency_pairs.append(pair)
        
        return list(set(currency_pairs))  # Remove duplicates
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from tweet text"""
        text_lower = text.lower()
        keywords = []
        
        # Economic indicators
        indicators = ['gdp', 'inflation', 'unemployment', 'interest rate', 'trade balance',
                     'current account', 'retail sales', 'manufacturing', 'services']
        
        for indicator in indicators:
            if indicator in text_lower:
                keywords.append(indicator)
        
        # Policy terms
        policy_terms = ['monetary policy', 'fiscal policy', 'quantitative easing', 'taper',
                       'dovish', 'hawkish', 'accommodative', 'restrictive']
        
        for term in policy_terms:
            if term in text_lower:
                keywords.append(term)
        
        # Central banks
        central_banks = ['fed', 'federal reserve', 'ecb', 'european central bank',
                        'boe', 'bank of england', 'boj', 'bank of japan',
                        'snb', 'swiss national bank', 'boc', 'bank of canada',
                        'rba', 'reserve bank of australia', 'rbnz', 'reserve bank of new zealand']
        
        for bank in central_banks:
            if bank in text_lower:
                keywords.append(bank)
        
        # Market sentiment
        sentiment_terms = ['bullish', 'bearish', 'positive', 'negative', 'optimistic', 'pessimistic',
                          'strong', 'weak', 'rise', 'fall', 'up', 'down', 'gain', 'loss']
        
        for term in sentiment_terms:
            if term in text_lower:
                keywords.append(term)
        
        return list(set(keywords))  # Remove duplicates
    
    async def get_trending_forex_tweets(self, limit: int = 50) -> List[NewsArticle]:
        """Get trending forex-related tweets"""
        # Use trending hashtags and keywords
        trending_keywords = [
            'forex', 'fx', 'EURUSD', 'GBPUSD', 'USDJPY', 'central bank',
            'interest rate', 'inflation', 'fed', 'ecb', 'boe'
        ]
        
        return await self.fetch_news(
            limit=limit,
            keywords=trending_keywords
        )
    
    async def get_verified_account_tweets(self, 
                                        account_usernames: List[str],
                                        limit: int = 50) -> List[NewsArticle]:
        """Get tweets from verified accounts"""
        articles = []
        
        for username in account_usernames:
            try:
                # Search for tweets from specific user
                query = f'from:{username} (forex OR fx OR currency OR "exchange rate" OR "central bank")'
                
                params = {
                    'query': query,
                    'max_results': min(limit // len(account_usernames), 100),
                    'tweet.fields': 'created_at,author_id,public_metrics,lang',
                    'user.fields': 'username,verified,public_metrics',
                    'expansions': 'author_id'
                }
                
                url = f"{self.base_url}/tweets/search/recent"
                response = await self._make_request('GET', url, params=params)
                
                data = response.json()
                if 'data' in data:
                    users = {}
                    if 'includes' in data and 'users' in data['includes']:
                        users = {user['id']: user for user in data['includes']['users']}
                    
                    for tweet_data in data['data']:
                        article = self._parse_tweet(tweet_data, users)
                        if article:
                            articles.append(article)
                
            except Exception as e:
                logger.error(f"Error fetching tweets from @{username}: {e}")
                continue
        
        logger.info(f"Fetched {len(articles)} tweets from verified accounts")
        return articles

