#!/usr/bin/env python3
"""
Lightweight performance test using only the Free Forex data source.
Avoids heavy processing/storage to run locally with minimal deps.
"""

import asyncio
import logging
from datetime import datetime

import os
import sys

# Ensure project root is in path
ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)
sys.path.append(ROOT)

from app1.src.data_ingestion.brokers.broker_manager import BrokerManager
from app1.src.data_ingestion.brokers.free_forex_client import FreeForexClient
from app1.src.data_ingestion.engines.ingestion_engine import DataIngestionEngine, IngestionConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    logger.info("🚀 Starting Free Source Performance Test")

    # Setup broker manager with only FreeForex
    manager = BrokerManager()
    free_client = FreeForexClient()
    await manager.add_broker("free_forex", free_client, priority=1)
    await manager.start_health_monitoring()

    # Configure ingestion engine (disable caching if desired)
    config = IngestionConfig(
        max_concurrent_requests=5,
        batch_size=50,
        flush_interval=0.5,
        enable_caching=False,
    )
    engine = DataIngestionEngine(manager, config)

    # Basic callbacks to avoid heavy processing
    async def on_price(price):
        # Minimal work; just count via metrics
        return

    engine.add_price_callback(on_price)

    await engine.start()
    try:
        instruments = ["EUR_USD", "GBP_USD"]
        logger.info(f"Starting price ingestion for {instruments}")
        await engine.ingest_prices(instruments)

        # Run for a short window
        start = datetime.now()
        duration_s = 6
        for _ in range(duration_s):
            await asyncio.sleep(1)
            metrics = engine.get_metrics()
            logger.info(
                f"Metrics: points={metrics['data_points_ingested']} "
                f"succ={metrics['successful_requests']} fail={metrics['failed_requests']} "
                f"avg_ms={metrics['avg_latency_ms']} max_ms={metrics['max_latency_ms']}"
            )

        final = engine.get_metrics()
        logger.info("\n=== Performance Summary ===")
        logger.info(f"Data points ingested: {final['data_points_ingested']}")
        logger.info(f"Success rate: {final['success_rate']:.1f}%")
        logger.info(f"Avg latency: {final['avg_latency_ms']} ms")
        logger.info(f"Max latency: {final['max_latency_ms']} ms")
        logger.info(f"Errors: {final['error_counts']}")
        logger.info("✅ Free Source Performance Test completed")

    finally:
        await engine.stop()
        await manager.stop_health_monitoring()


if __name__ == "__main__":
    asyncio.run(main())


