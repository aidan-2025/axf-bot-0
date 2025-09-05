#!/usr/bin/env python3
"""
News Processor
Processes and enriches news articles and economic events
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .models import NewsArticle, EconomicEvent, NewsIngestionConfig, NewsSource, Currency

logger = logging.getLogger(__name__)

class NewsProcessor:
    """Processes and enriches news articles and economic events"""
    
    def __init__(self, config: NewsIngestionConfig):
        """Initialize news processor"""
        self.config = config
        
        # Currency pair patterns
        self.currency_pair_patterns = [
            r'\b([A-Z]{3})/([A-Z]{3})\b',  # EUR/USD
            r'\b([A-Z]{3})([A-Z]{3})\b',   # EURUSD
            r'\b([A-Z]{3})\s*([A-Z]{3})\b' # EUR USD
        ]
        
        # Economic indicator patterns
        self.economic_indicators = {
            'gdp': r'\b(gdp|gross domestic product)\b',
            'inflation': r'\b(inflation|cpi|consumer price index)\b',
            'unemployment': r'\b(unemployment|jobless|employment)\b',
            'interest_rate': r'\b(interest rate|rate|fed funds|policy rate)\b',
            'trade_balance': r'\b(trade balance|trade deficit|trade surplus)\b',
            'current_account': r'\b(current account|balance of payments)\b',
            'retail_sales': r'\b(retail sales|consumer spending)\b',
            'manufacturing': r'\b(manufacturing|pmi|industrial production)\b',
            'services': r'\b(services|service sector)\b',
            'consumer_confidence': r'\b(consumer confidence|consumer sentiment)\b'
        }
        
        # Central bank patterns
        self.central_banks = {
            'fed': r'\b(fed|federal reserve|fomc)\b',
            'ecb': r'\b(ecb|european central bank)\b',
            'boe': r'\b(boe|bank of england|mpc)\b',
            'boj': r'\b(boj|bank of japan)\b',
            'snb': r'\b(snb|swiss national bank)\b',
            'boc': r'\b(boc|bank of canada)\b',
            'rba': r'\b(rba|reserve bank of australia)\b',
            'rbnz': r'\b(rbnz|reserve bank of new zealand)\b'
        }
        
        # Policy term patterns
        self.policy_terms = {
            'monetary_policy': r'\b(monetary policy|monetary stance)\b',
            'fiscal_policy': r'\b(fiscal policy|fiscal stance|budget)\b',
            'quantitative_easing': r'\b(quantitative easing|qe|asset purchases)\b',
            'taper': r'\b(taper|tapering|taper tantrum)\b',
            'dovish': r'\b(dovish|dove|accommodative)\b',
            'hawkish': r'\b(hawkish|hawk|restrictive)\b'
        }
    
    async def process_article(self, article: NewsArticle) -> Optional[NewsArticle]:
        """Process and enrich a news article"""
        try:
            # Create a copy to avoid modifying the original
            processed_article = NewsArticle(
                article_id=article.article_id,
                title=article.title,
                content=article.content,
                summary=article.summary,
                source=article.source,
                author=article.author,
                published_at=article.published_at,
                language=article.language,
                url=article.url,
                currency_pairs=article.currency_pairs.copy(),
                keywords=article.keywords.copy(),
                relevance_score=article.relevance_score,
                sentiment_label=article.sentiment_label,
                sentiment_score=article.sentiment_score,
                confidence=article.confidence,
                raw_data=article.raw_data.copy(),
                created_at=article.created_at,
                updated_at=article.updated_at,
                content_hash=article.content_hash
            )
            
            # Extract currency pairs
            extracted_pairs = self._extract_currency_pairs(processed_article.title, processed_article.content)
            processed_article.currency_pairs.extend(extracted_pairs)
            processed_article.currency_pairs = list(set(processed_article.currency_pairs))  # Remove duplicates
            
            # Extract keywords
            extracted_keywords = self._extract_keywords(processed_article.title, processed_article.content)
            processed_article.keywords.extend(extracted_keywords)
            processed_article.keywords = list(set(processed_article.keywords))  # Remove duplicates
            
            # Calculate relevance score
            processed_article.relevance_score = self._calculate_relevance_score(
                processed_article.title,
                processed_article.content,
                processed_article.currency_pairs,
                processed_article.keywords
            )
            
            # Detect language if not set
            if not processed_article.language or processed_article.language == 'auto':
                processed_article.language = self._detect_language(processed_article.content)
            
            # Generate summary if not provided
            if not processed_article.summary:
                processed_article.summary = self._generate_summary(processed_article.content)
            
            # Update timestamp
            processed_article.updated_at = datetime.utcnow()
            
            logger.debug(f"Processed article: {processed_article.article_id}")
            return processed_article
            
        except Exception as e:
            logger.error(f"Error processing article {article.article_id}: {e}")
            return None
    
    async def process_event(self, event: EconomicEvent) -> Optional[EconomicEvent]:
        """Process and enrich an economic event"""
        try:
            # Create a copy to avoid modifying the original
            processed_event = EconomicEvent(
                event_id=event.event_id,
                title=event.title,
                description=event.description,
                event_time=event.event_time,
                timezone=event.timezone,
                impact=event.impact,
                currency=event.currency,
                currency_pairs=event.currency_pairs.copy(),
                actual=event.actual,
                forecast=event.forecast,
                previous=event.previous,
                unit=event.unit,
                source=event.source,
                country=event.country,
                category=event.category,
                created_at=event.created_at,
                updated_at=event.updated_at
            )
            
            # Extract currency pairs from title and description
            if processed_event.description:
                extracted_pairs = self._extract_currency_pairs(processed_event.title, processed_event.description)
                processed_event.currency_pairs.extend(extracted_pairs)
                processed_event.currency_pairs = list(set(processed_event.currency_pairs))  # Remove duplicates
            
            # Determine currency pairs based on currency
            currency_pairs = self._get_currency_pairs(processed_event.currency)
            processed_event.currency_pairs.extend(currency_pairs)
            processed_event.currency_pairs = list(set(processed_event.currency_pairs))  # Remove duplicates
            
            # Update timestamp
            processed_event.updated_at = datetime.utcnow()
            
            logger.debug(f"Processed event: {processed_event.event_id}")
            return processed_event
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            return None
    
    def _extract_currency_pairs(self, title: str, content: str) -> List[str]:
        """Extract currency pairs from text"""
        text = f"{title} {content}".upper()
        pairs = []
        
        for pattern in self.currency_pair_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) == 2:
                    pair = f"{match[0]}{match[1]}"
                    if self._is_valid_currency_pair(pair):
                        pairs.append(pair)
        
        return list(set(pairs))  # Remove duplicates
    
    def _is_valid_currency_pair(self, pair: str) -> bool:
        """Check if a currency pair is valid"""
        if len(pair) != 6:
            return False
        
        base = pair[:3]
        quote = pair[3:]
        
        # Check if both currencies are valid
        try:
            Currency(base)
            Currency(quote)
            return True
        except ValueError:
            return False
    
    def _extract_keywords(self, title: str, content: str) -> List[str]:
        """Extract relevant keywords from text"""
        text = f"{title} {content}".lower()
        keywords = []
        
        # Extract economic indicators
        for indicator, pattern in self.economic_indicators.items():
            if re.search(pattern, text, re.IGNORECASE):
                keywords.append(indicator)
        
        # Extract central banks
        for bank, pattern in self.central_banks.items():
            if re.search(pattern, text, re.IGNORECASE):
                keywords.append(bank)
        
        # Extract policy terms
        for term, pattern in self.policy_terms.items():
            if re.search(pattern, text, re.IGNORECASE):
                keywords.append(term)
        
        # Extract forex-specific terms
        forex_terms = [
            'forex', 'fx', 'currency', 'exchange rate', 'foreign exchange',
            'pip', 'spread', 'leverage', 'margin', 'position', 'trade',
            'bullish', 'bearish', 'long', 'short', 'buy', 'sell'
        ]
        
        for term in forex_terms:
            if term in text:
                keywords.append(term)
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_relevance_score(self, 
                                 title: str, 
                                 content: str, 
                                 currency_pairs: List[str],
                                 keywords: List[str]) -> float:
        """Calculate relevance score for forex trading"""
        score = 0.0
        text = f"{title} {content}".lower()
        
        # Base score from currency pairs
        if currency_pairs:
            score += min(len(currency_pairs) * 0.1, 0.3)
        
        # Score from keywords
        keyword_weights = {
            'forex': 0.2, 'fx': 0.2, 'currency': 0.15, 'exchange rate': 0.15,
            'central bank': 0.25, 'interest rate': 0.2, 'inflation': 0.15,
            'gdp': 0.1, 'unemployment': 0.1, 'trade balance': 0.1,
            'monetary policy': 0.2, 'fiscal policy': 0.15,
            'quantitative easing': 0.15, 'taper': 0.1,
            'dovish': 0.1, 'hawkish': 0.1
        }
        
        for keyword in keywords:
            weight = keyword_weights.get(keyword, 0.05)
            score += weight
        
        # Score from economic indicators
        for indicator, pattern in self.economic_indicators.items():
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1
        
        # Score from central banks
        for bank, pattern in self.central_banks.items():
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.15
        
        # Score from policy terms
        for term, pattern in self.policy_terms.items():
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.1
        
        # Normalize to 0-1 range
        return min(score, 1.0)
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text"""
        # Simple language detection based on character patterns
        if re.search(r'[а-яё]', text, re.IGNORECASE):
            return 'ru'
        elif re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return 'ja'
        elif re.search(r'[ñáéíóúü]', text, re.IGNORECASE):
            return 'es'
        elif re.search(r'[àâäéèêëïîôöùûüÿç]', text, re.IGNORECASE):
            return 'fr'
        elif re.search(r'[äöüß]', text, re.IGNORECASE):
            return 'de'
        elif re.search(r'[àèéìíîòóùú]', text, re.IGNORECASE):
            return 'it'
        elif re.search(r'[ãõáéíóúâêô]', text, re.IGNORECASE):
            return 'pt'
        else:
            return 'en'  # Default to English
    
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
    
    async def process_batch_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Process a batch of articles"""
        processed_articles = []
        
        for article in articles:
            processed_article = await self.process_article(article)
            if processed_article:
                processed_articles.append(processed_article)
        
        return processed_articles
    
    async def process_batch_events(self, events: List[EconomicEvent]) -> List[EconomicEvent]:
        """Process a batch of events"""
        processed_events = []
        
        for event in events:
            processed_event = await self.process_event(event)
            if processed_event:
                processed_events.append(processed_event)
        
        return processed_events

