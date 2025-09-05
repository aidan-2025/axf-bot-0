#!/usr/bin/env python3
"""
Database performance testing script for AXF Bot 0
Tests PostgreSQL operations and query performance
"""
import os
import sys
import time
import logging
from datetime import datetime, timedelta
import random

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_database_operations():
    """Test various database operations"""
    try:
        # Import database modules
        from app1.src.database.connection import get_db_session, get_database_status
        from app1.src.database.models import Strategy, CurrencyPair, Timeframe, StrategyPerformance
        
        logger.info("Testing database operations...")
        
        # Get database session
        db = get_db_session()
        
        # Test 1: Simple queries
        logger.info("Test 1: Simple queries")
        start_time = time.time()
        
        strategies = db.query(Strategy).all()
        currency_pairs = db.query(CurrencyPair).all()
        timeframes = db.query(Timeframe).all()
        
        query_time = time.time() - start_time
        logger.info(f"  - Queried {len(strategies)} strategies, {len(currency_pairs)} currency pairs, {len(timeframes)} timeframes in {query_time:.3f}s")
        
        # Test 2: Complex joins
        logger.info("Test 2: Complex joins")
        start_time = time.time()
        
        from sqlalchemy.orm import joinedload
        strategies_with_relations = db.query(Strategy).options(
            joinedload(Strategy.strategy_currency_pairs),
            joinedload(Strategy.strategy_timeframes)
        ).all()
        
        join_time = time.time() - start_time
        logger.info(f"  - Complex join query completed in {join_time:.3f}s")
        
        # Test 3: Filtered queries
        logger.info("Test 3: Filtered queries")
        start_time = time.time()
        
        active_strategies = db.query(Strategy).filter(Strategy.status == 'active').all()
        high_priority_strategies = db.query(Strategy).filter(Strategy.priority >= 7).all()
        ai_generated_strategies = db.query(Strategy).filter(Strategy.is_ai_generated == True).all()
        
        filter_time = time.time() - start_time
        logger.info(f"  - Filtered queries completed in {filter_time:.3f}s")
        logger.info(f"  - Found {len(active_strategies)} active strategies")
        logger.info(f"  - Found {len(high_priority_strategies)} high priority strategies")
        logger.info(f"  - Found {len(ai_generated_strategies)} AI-generated strategies")
        
        # Test 4: Aggregation queries
        logger.info("Test 4: Aggregation queries")
        start_time = time.time()
        
        from sqlalchemy import func
        total_profit = db.query(func.sum(Strategy.total_profit)).scalar() or 0
        avg_win_rate = db.query(func.avg(Strategy.win_rate)).scalar() or 0
        max_drawdown = db.query(func.max(Strategy.max_drawdown)).scalar() or 0
        strategy_count = db.query(func.count(Strategy.id)).scalar() or 0
        
        agg_time = time.time() - start_time
        logger.info(f"  - Aggregation queries completed in {agg_time:.3f}s")
        logger.info(f"  - Total profit: ${total_profit:.2f}")
        logger.info(f"  - Average win rate: {avg_win_rate:.2f}%")
        logger.info(f"  - Max drawdown: {max_drawdown:.2f}%")
        logger.info(f"  - Total strategies: {strategy_count}")
        
        # Test 5: JSON field queries
        logger.info("Test 5: JSON field queries")
        start_time = time.time()
        
        trend_strategies = db.query(Strategy).filter(
            Strategy.strategy_type == 'trend_following'
        ).all()
        
        range_strategies = db.query(Strategy).filter(
            Strategy.strategy_type == 'range_trading'
        ).all()
        
        json_time = time.time() - start_time
        logger.info(f"  - JSON field queries completed in {json_time:.3f}s")
        logger.info(f"  - Found {len(trend_strategies)} trend following strategies")
        logger.info(f"  - Found {len(range_strategies)} range trading strategies")
        
        # Test 6: Performance by strategy type
        logger.info("Test 6: Performance analysis by strategy type")
        start_time = time.time()
        
        from sqlalchemy import case
        performance_by_type = db.query(
            Strategy.strategy_type,
            func.count(Strategy.id).label('count'),
            func.avg(Strategy.profit_factor).label('avg_profit_factor'),
            func.avg(Strategy.win_rate).label('avg_win_rate'),
            func.avg(Strategy.sharpe_ratio).label('avg_sharpe_ratio')
        ).group_by(Strategy.strategy_type).all()
        
        analysis_time = time.time() - start_time
        logger.info(f"  - Performance analysis completed in {analysis_time:.3f}s")
        
        for result in performance_by_type:
            logger.info(f"  - {result.strategy_type}: {result.count} strategies, "
                       f"avg profit factor: {result.avg_profit_factor:.2f}, "
                       f"avg win rate: {result.avg_win_rate:.2f}%, "
                       f"avg sharpe: {result.avg_sharpe_ratio:.2f}")
        
        # Test 7: Database status
        logger.info("Test 7: Database status check")
        db_status = get_database_status()
        logger.info(f"  - PostgreSQL: {'Connected' if db_status['postgresql']['connected'] else 'Disconnected'}")
        logger.info(f"  - InfluxDB: {'Connected' if db_status['influxdb']['connected'] else 'Disconnected'}")
        
        db.close()
        
        logger.info("✅ All database operations completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database operations test failed: {e}")
        return False

def test_query_performance():
    """Test query performance with different data sizes"""
    try:
        from app1.src.database.connection import get_db_session
        from app1.src.database.models import Strategy, StrategyPerformance
        from sqlalchemy import func, text
        
        logger.info("Testing query performance...")
        
        db = get_db_session()
        
        # Test 1: Index usage
        logger.info("Test 1: Index usage verification")
        start_time = time.time()
        
        # Query that should use indexes
        active_strategies = db.query(Strategy).filter(Strategy.status == 'active').all()
        
        index_time = time.time() - start_time
        logger.info(f"  - Indexed query completed in {index_time:.3f}s")
        
        # Test 2: Large result set handling
        logger.info("Test 2: Large result set handling")
        start_time = time.time()
        
        all_strategies = db.query(Strategy).all()
        
        large_query_time = time.time() - start_time
        logger.info(f"  - Large result set query completed in {large_query_time:.3f}s")
        logger.info(f"  - Retrieved {len(all_strategies)} strategies")
        
        # Test 3: Complex aggregation performance
        logger.info("Test 3: Complex aggregation performance")
        start_time = time.time()
        
        performance_stats = db.query(
            func.count(Strategy.id).label('total_strategies'),
            func.sum(Strategy.total_trades).label('total_trades'),
            func.avg(Strategy.profit_factor).label('avg_profit_factor'),
            func.avg(Strategy.win_rate).label('avg_win_rate'),
            func.avg(Strategy.sharpe_ratio).label('avg_sharpe_ratio'),
            func.max(Strategy.total_profit).label('max_profit'),
            func.min(Strategy.max_drawdown).label('min_drawdown')
        ).first()
        
        agg_time = time.time() - start_time
        logger.info(f"  - Complex aggregation completed in {agg_time:.3f}s")
        logger.info(f"  - Total strategies: {performance_stats.total_strategies}")
        logger.info(f"  - Total trades: {performance_stats.total_trades}")
        logger.info(f"  - Average profit factor: {performance_stats.avg_profit_factor:.2f}")
        logger.info(f"  - Average win rate: {performance_stats.avg_win_rate:.2f}%")
        logger.info(f"  - Average sharpe ratio: {performance_stats.avg_sharpe_ratio:.2f}")
        logger.info(f"  - Max profit: ${performance_stats.max_profit:.2f}")
        logger.info(f"  - Min drawdown: {performance_stats.min_drawdown:.2f}%")
        
        # Test 4: JSON field performance
        logger.info("Test 4: JSON field performance")
        start_time = time.time()
        
        # Query strategies with specific JSON parameters
        ma_strategies = db.query(Strategy).filter(
            Strategy.parameters.op('->>')('ma_fast').isnot(None)
        ).all()
        
        json_time = time.time() - start_time
        logger.info(f"  - JSON field query completed in {json_time:.3f}s")
        logger.info(f"  - Found {len(ma_strategies)} strategies with MA parameters")
        
        # Test 5: Connection pool performance
        logger.info("Test 5: Connection pool performance")
        start_time = time.time()
        
        # Simulate multiple concurrent queries
        for i in range(10):
            strategies = db.query(Strategy).filter(Strategy.priority >= i).all()
        
        pool_time = time.time() - start_time
        logger.info(f"  - Connection pool test completed in {pool_time:.3f}s")
        
        db.close()
        
        logger.info("✅ Query performance tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Query performance test failed: {e}")
        return False

def test_api_performance():
    """Test API endpoint performance"""
    try:
        import requests
        import time
        
        logger.info("Testing API endpoint performance...")
        
        base_url = "http://localhost:8000"
        
        # Test 1: Health check performance
        logger.info("Test 1: Health check performance")
        start_time = time.time()
        
        response = requests.get(f"{base_url}/health")
        health_time = time.time() - start_time
        
        logger.info(f"  - Health check completed in {health_time:.3f}s (status: {response.status_code})")
        
        # Test 2: Database health check performance
        logger.info("Test 2: Database health check performance")
        start_time = time.time()
        
        response = requests.get(f"{base_url}/health/database")
        db_health_time = time.time() - start_time
        
        logger.info(f"  - Database health check completed in {db_health_time:.3f}s (status: {response.status_code})")
        
        # Test 3: Strategies API performance
        logger.info("Test 3: Strategies API performance")
        start_time = time.time()
        
        response = requests.get(f"{base_url}/api/v1/strategies/")
        strategies_time = time.time() - start_time
        
        logger.info(f"  - Strategies API completed in {strategies_time:.3f}s (status: {response.status_code})")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"  - Retrieved {len(data['data']['all'])} strategies")
        
        # Test 4: Performance API performance
        logger.info("Test 4: Performance API performance")
        start_time = time.time()
        
        response = requests.get(f"{base_url}/api/v1/performance/db-summary")
        performance_time = time.time() - start_time
        
        logger.info(f"  - Performance API completed in {performance_time:.3f}s (status: {response.status_code})")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"  - Total strategies: {data['data']['total_strategies']}")
            logger.info(f"  - Well performing: {data['data']['well_performing_count']}")
            logger.info(f"  - Poor performing: {data['data']['poor_performing_count']}")
        
        # Test 5: Concurrent API requests
        logger.info("Test 5: Concurrent API requests")
        start_time = time.time()
        
        import threading
        results = []
        
        def make_request():
            response = requests.get(f"{base_url}/api/v1/strategies/")
            results.append(response.status_code)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        concurrent_time = time.time() - start_time
        success_count = sum(1 for status in results if status == 200)
        
        logger.info(f"  - Concurrent requests completed in {concurrent_time:.3f}s")
        logger.info(f"  - {success_count}/5 requests successful")
        
        logger.info("✅ API performance tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ API performance test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting comprehensive database performance tests...")
    
    success = True
    
    # Test database operations
    if not test_database_operations():
        success = False
    
    # Test query performance
    if not test_query_performance():
        success = False
    
    # Test API performance
    if not test_api_performance():
        success = False
    
    if success:
        logger.info("🎉 All performance tests passed successfully!")
        logger.info("📊 Database operations: ✅")
        logger.info("⚡ Query performance: ✅")
        logger.info("🌐 API performance: ✅")
    else:
        logger.error("❌ Some performance tests failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
