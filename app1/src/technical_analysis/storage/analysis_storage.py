"""
Storage for technical analysis results in InfluxDB
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

from ..models import AnalysisResult, MarketAnalysis, TechnicalIndicator, Timeframe

logger = logging.getLogger(__name__)


class AnalysisStorage:
    """Storage for technical analysis results in InfluxDB"""
    
    def __init__(self, influxdb_config: Dict[str, str]):
        self.config = influxdb_config
        self.client = None
        self.write_api = None
        self.query_api = None
        self.bucket = influxdb_config.get('bucket', 'technical_analysis')
        
    def connect(self) -> bool:
        """Connect to InfluxDB"""
        try:
            self.client = InfluxDBClient(
                url=self.config['url'],
                token=self.config['token'],
                org=self.config['org']
            )
            
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            
            # Test connection
            self.client.ping()
            logger.info("Connected to InfluxDB for technical analysis storage")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from InfluxDB"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from InfluxDB")
    
    def store_analysis_result(self, analysis_result: AnalysisResult) -> bool:
        """Store complete analysis result"""
        try:
            if not self.write_api:
                logger.error("Not connected to InfluxDB")
                return False
            
            points = []
            
            # Store analysis metadata
            metadata_point = Point("analysis_metadata") \
                .tag("symbol", analysis_result.symbol) \
                .tag("success", str(analysis_result.success)) \
                .field("timeframes_count", len(analysis_result.analyses)) \
                .field("processing_time_ms", analysis_result.processing_time_ms) \
                .field("calculated_at", analysis_result.calculated_at.isoformat()) \
                .time(analysis_result.calculated_at)
            
            if analysis_result.error_message:
                metadata_point.field("error_message", analysis_result.error_message)
            
            points.append(metadata_point)
            
            # Store individual timeframe analyses
            for timeframe, analysis in analysis_result.analyses.items():
                timeframe_points = self._create_timeframe_points(analysis)
                points.extend(timeframe_points)
            
            # Write all points
            self.write_api.write(bucket=self.bucket, record=points)
            
            logger.info(f"Stored analysis result for {analysis_result.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing analysis result: {str(e)}")
            return False
    
    def _create_timeframe_points(self, analysis: MarketAnalysis) -> List[Point]:
        """Create InfluxDB points for a timeframe analysis"""
        points = []
        
        try:
            # Store timeframe metadata
            timeframe_point = Point("timeframe_analysis") \
                .tag("symbol", analysis.symbol) \
                .tag("timeframe", analysis.timeframe.value) \
                .field("data_points", len(analysis.price_data)) \
                .field("indicators_count", len(analysis.indicators)) \
                .field("analysis_timestamp", analysis.timestamp.isoformat()) \
                .time(analysis.timestamp)
            
            # Add metadata fields
            for key, value in analysis.analysis_metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    timeframe_point.field(f"meta_{key}", value)
                else:
                    timeframe_point.field(f"meta_{key}", json.dumps(value))
            
            points.append(timeframe_point)
            
            # Store individual indicators
            for indicator_name, indicator in analysis.indicators.items():
                indicator_points = self._create_indicator_points(indicator)
                points.extend(indicator_points)
            
            return points
            
        except Exception as e:
            logger.error(f"Error creating timeframe points: {str(e)}")
            return []
    
    def _create_indicator_points(self, indicator: TechnicalIndicator) -> List[Point]:
        """Create InfluxDB points for a technical indicator"""
        points = []
        
        try:
            # Store indicator metadata
            metadata_point = Point("indicator_metadata") \
                .tag("symbol", indicator.symbol) \
                .tag("timeframe", indicator.timeframe.value) \
                .tag("indicator_name", indicator.name) \
                .field("values_count", len(indicator.values)) \
                .field("calculated_at", indicator.calculated_at.isoformat()) \
                .time(indicator.calculated_at)
            
            # Add parameter fields
            for key, value in indicator.parameters.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata_point.field(f"param_{key}", value)
                else:
                    metadata_point.field(f"param_{key}", json.dumps(value))
            
            points.append(metadata_point)
            
            # Store indicator values (time series data)
            for i, (value, timestamp) in enumerate(zip(indicator.values, indicator.timestamps)):
                if value is not None and not (isinstance(value, float) and (value != value)):  # Check for NaN
                    value_point = Point("indicator_values") \
                        .tag("symbol", indicator.symbol) \
                        .tag("timeframe", indicator.timeframe.value) \
                        .tag("indicator_name", indicator.name) \
                        .field("value", float(value)) \
                        .field("index", i) \
                        .time(timestamp)
                    
                    points.append(value_point)
            
            return points
            
        except Exception as e:
            logger.error(f"Error creating indicator points: {str(e)}")
            return []
    
    def get_analysis_results(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[Timeframe] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query analysis results from InfluxDB"""
        try:
            if not self.query_api:
                logger.error("Not connected to InfluxDB")
                return []
            
            # Build query
            query = f'from(bucket: "{self.bucket}")'
            query += ' |> range(start: -30d)'
            
            if start_time:
                query += f' |> range(start: {start_time.isoformat()})'
            if end_time:
                query += f' |> range(stop: {end_time.isoformat()})'
            
            query += ' |> filter(fn: (r) => r._measurement == "analysis_metadata")'
            
            if symbol:
                query += f' |> filter(fn: (r) => r.symbol == "{symbol}")'
            
            query += f' |> limit(n: {limit})'
            
            # Execute query
            result = self.query_api.query(query)
            
            # Process results
            results = []
            for table in result:
                for record in table.records:
                    result_dict = {
                        'symbol': record.values.get('symbol'),
                        'success': record.values.get('success') == 'true',
                        'timeframes_count': record.values.get('timeframes_count'),
                        'processing_time_ms': record.values.get('processing_time_ms'),
                        'calculated_at': record.values.get('calculated_at'),
                        'timestamp': record.get_time()
                    }
                    
                    if record.values.get('error_message'):
                        result_dict['error_message'] = record.values.get('error_message')
                    
                    results.append(result_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying analysis results: {str(e)}")
            return []
    
    def get_indicator_values(
        self,
        symbol: str,
        indicator_name: str,
        timeframe: Timeframe,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Query indicator values from InfluxDB"""
        try:
            if not self.query_api:
                logger.error("Not connected to InfluxDB")
                return []
            
            # Build query
            query = f'from(bucket: "{self.bucket}")'
            query += ' |> range(start: -7d)'
            
            if start_time:
                query += f' |> range(start: {start_time.isoformat()})'
            if end_time:
                query += f' |> range(stop: {end_time.isoformat()})'
            
            query += ' |> filter(fn: (r) => r._measurement == "indicator_values")'
            query += f' |> filter(fn: (r) => r.symbol == "{symbol}")'
            query += f' |> filter(fn: (r) => r.indicator_name == "{indicator_name}")'
            query += f' |> filter(fn: (r) => r.timeframe == "{timeframe.value}")'
            query += f' |> limit(n: {limit})'
            query += ' |> sort(columns: ["_time"])'
            
            # Execute query
            result = self.query_api.query(query)
            
            # Process results
            values = []
            for table in result:
                for record in table.records:
                    value_dict = {
                        'timestamp': record.get_time(),
                        'value': record.values.get('_value'),
                        'index': record.values.get('index')
                    }
                    values.append(value_dict)
            
            return values
            
        except Exception as e:
            logger.error(f"Error querying indicator values: {str(e)}")
            return []
    
    def get_analysis_summary(
        self,
        symbol: Optional[str] = None,
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """Get analysis summary statistics"""
        try:
            if not self.query_api:
                logger.error("Not connected to InfluxDB")
                return {}
            
            start_time = datetime.now() - timedelta(hours=hours_back)
            
            # Query analysis metadata
            query = f'''
            from(bucket: "{self.bucket}")
            |> range(start: {start_time.isoformat()})
            |> filter(fn: (r) => r._measurement == "analysis_metadata")
            '''
            
            if symbol:
                query += f' |> filter(fn: (r) => r.symbol == "{symbol}")'
            
            result = self.query_api.query(query)
            
            # Process results
            total_analyses = 0
            successful_analyses = 0
            total_processing_time = 0
            symbols_analyzed = set()
            
            for table in result:
                for record in table.records:
                    total_analyses += 1
                    if record.values.get('success') == 'true':
                        successful_analyses += 1
                    if record.values.get('processing_time_ms'):
                        total_processing_time += record.values.get('processing_time_ms')
                    if record.values.get('symbol'):
                        symbols_analyzed.add(record.values.get('symbol'))
            
            return {
                'total_analyses': total_analyses,
                'successful_analyses': successful_analyses,
                'success_rate': (successful_analyses / total_analyses * 100) if total_analyses > 0 else 0,
                'average_processing_time_ms': (total_processing_time / total_analyses) if total_analyses > 0 else 0,
                'symbols_analyzed': list(symbols_analyzed),
                'time_period_hours': hours_back
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis summary: {str(e)}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> bool:
        """Clean up old analysis data"""
        try:
            if not self.query_api:
                logger.error("Not connected to InfluxDB")
                return False
            
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            
            # Delete old data
            delete_query = f'''
            from(bucket: "{self.bucket}")
            |> range(start: 1970-01-01T00:00:00Z, stop: {cutoff_time.isoformat()})
            |> delete()
            '''
            
            # Note: This requires delete permissions in InfluxDB
            # For now, just log the query
            logger.info(f"Would delete data older than {cutoff_time.isoformat()}")
            logger.info(f"Delete query: {delete_query}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")
            return False

