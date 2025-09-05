#!/usr/bin/env python3
"""
Batch Processor

Handles batch processing of multiple strategies with resource management.
"""

import asyncio
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import logging
import json
import psutil
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    
    # Processing settings
    max_workers: int = 4
    memory_limit_mb: int = 2048
    timeout_seconds: int = 300
    batch_size: int = 10
    
    # Resource management
    memory_check_interval: int = 5  # seconds
    max_memory_usage: float = 0.8  # 80% of available memory
    cleanup_interval: int = 10  # seconds
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    
    # Progress tracking
    progress_callback: Optional[Callable] = None
    log_interval: int = 10  # log every N processed items


class BatchProcessor:
    """Handles batch processing with resource management"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Processing state
        self.is_processing = False
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = None
        
        self.logger.info("BatchProcessor initialized")
    
    async def process_batch_async(self, items: List[Any], 
                                process_func: Callable) -> List[Dict[str, Any]]:
        """Process a batch of items asynchronously"""
        self.logger.info(f"Starting async batch processing of {len(items)} items")
        
        self.is_processing = True
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = datetime.now()
        
        try:
            # Split items into batches
            batches = self._split_into_batches(items, self.config.batch_size)
            
            results = []
            for batch_idx, batch in enumerate(batches):
                self.logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} items)")
                
                # Process batch
                batch_results = await self._process_batch_async(batch, process_func)
                results.extend(batch_results)
                
                # Check memory usage
                if self._check_memory_usage():
                    self.logger.warning("High memory usage detected, performing cleanup")
                    await self._cleanup_memory()
                
                # Progress callback
                if self.config.progress_callback:
                    self.config.progress_callback(
                        batch_idx + 1, 
                        len(batches), 
                        self.processed_count, 
                        self.failed_count
                    )
            
            # Final summary
            total_duration = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(f"Batch processing completed: {self.processed_count} processed, {self.failed_count} failed")
            self.logger.info(f"Total duration: {total_duration:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise
        finally:
            self.is_processing = False
    
    def process_batch_multiprocess(self, items: List[Any], 
                                 process_func: Callable) -> List[Dict[str, Any]]:
        """Process a batch of items using multiprocessing"""
        self.logger.info(f"Starting multiprocess batch processing of {len(items)} items")
        
        self.is_processing = True
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = datetime.now()
        
        try:
            # Split items into batches
            batches = self._split_into_batches(items, self.config.batch_size)
            
            results = []
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(self._process_batch_worker, batch, process_func): batch
                    for batch in batches
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_batch, timeout=self.config.timeout_seconds):
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                        
                        # Update counters
                        self.processed_count += len([r for r in batch_results if r.get('success', False)])
                        self.failed_count += len([r for r in batch_results if not r.get('success', False)])
                        
                        # Progress callback
                        if self.config.progress_callback:
                            batch_idx = list(future_to_batch.keys()).index(future)
                            self.config.progress_callback(
                                batch_idx + 1,
                                len(batches),
                                self.processed_count,
                                self.failed_count
                            )
                        
                    except Exception as e:
                        batch = future_to_batch[future]
                        self.logger.error(f"Batch processing failed: {e}")
                        self.failed_count += len(batch)
            
            # Final summary
            total_duration = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(f"Multiprocess batch processing completed: {self.processed_count} processed, {self.failed_count} failed")
            self.logger.info(f"Total duration: {total_duration:.2f} seconds")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multiprocess batch processing failed: {e}")
            raise
        finally:
            self.is_processing = False
    
    async def _process_batch_async(self, batch: List[Any], 
                                 process_func: Callable) -> List[Dict[str, Any]]:
        """Process a single batch asynchronously"""
        tasks = []
        for item in batch:
            task = self._process_item_with_retry(item, process_func)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Item {i} failed with exception: {result}")
                processed_results.append({
                    'success': False,
                    'item': batch[i],
                    'error': str(result),
                    'duration_seconds': 0
                })
                self.failed_count += 1
            else:
                processed_results.append(result)
                if result.get('success', False):
                    self.processed_count += 1
                else:
                    self.failed_count += 1
        
        return processed_results
    
    def _process_batch_worker(self, batch: List[Any], 
                            process_func: Callable) -> List[Dict[str, Any]]:
        """Worker function for multiprocessing"""
        results = []
        
        for item in batch:
            try:
                # This would call the actual process function
                # For now, return a mock result
                result = {
                    'success': True,
                    'item': item,
                    'duration_seconds': 1.0
                }
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Item processing failed: {e}")
                results.append({
                    'success': False,
                    'item': item,
                    'error': str(e),
                    'duration_seconds': 0
                })
        
        return results
    
    async def _process_item_with_retry(self, item: Any, 
                                     process_func: Callable) -> Dict[str, Any]:
        """Process a single item with retry logic"""
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Call the process function
                result = await process_func(item)
                return result
                
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    self.logger.warning(f"Attempt {attempt + 1} failed for item {item}, retrying: {e}")
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    self.logger.error(f"All retry attempts failed for item {item}: {e}")
        
        return {
            'success': False,
            'item': item,
            'error': str(last_error),
            'duration_seconds': 0
        }
    
    def _split_into_batches(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """Split items into batches of specified size"""
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is too high"""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            return memory_percent > self.config.max_memory_usage
        except Exception:
            return False
    
    async def _cleanup_memory(self):
        """Perform memory cleanup"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Log memory usage
            memory_info = psutil.virtual_memory()
            self.logger.info(f"Memory cleanup completed. Usage: {memory_info.percent}%")
            
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        status = {
            'is_processing': self.is_processing,
            'processed_count': self.processed_count,
            'failed_count': self.failed_count,
            'total_count': self.processed_count + self.failed_count,
            'success_rate': 0.0
        }
        
        if status['total_count'] > 0:
            status['success_rate'] = self.processed_count / status['total_count']
        
        if self.start_time:
            status['duration_seconds'] = (datetime.now() - self.start_time).total_seconds()
        
        # Memory usage
        try:
            memory_info = psutil.virtual_memory()
            status['memory_usage_percent'] = memory_info.percent
            status['memory_available_mb'] = memory_info.available / (1024 * 1024)
        except Exception:
            status['memory_usage_percent'] = 0
            status['memory_available_mb'] = 0
        
        return status
    
    def reset_status(self):
        """Reset processing status"""
        self.is_processing = False
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = None
        self.logger.info("Processing status reset")

