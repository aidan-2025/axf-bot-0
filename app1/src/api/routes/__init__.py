"""
API Routes Package
"""

from .strategy import router as strategy_router
from .data import router as data_router
from .sentiment import router as sentiment_router
from .performance import router as performance_router
from .ai import router as ai_router
from .orchestrator import router as orchestrator_router
from .tools import router as tools_router
from .news import router as news_router
from .sentiment_analysis import router as sentiment_analysis_router
from .economic_calendar import router as economic_calendar_router
from .backtesting import router as backtesting_router
from .strategy_validation import router as strategy_validation_router
from .workflow import router as workflow_router

__all__ = ['strategy_router', 'data_router', 'sentiment_router', 'performance_router', 'ai_router', 'orchestrator_router', 'tools_router', 'news_router', 'sentiment_analysis_router', 'economic_calendar_router', 'backtesting_router', 'strategy_validation_router', 'workflow_router']
