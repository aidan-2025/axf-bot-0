"""
Validation Criteria

Defines validation criteria for strategy filtering and evaluation.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation level for criteria"""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class ValidationType(Enum):
    """Type of validation"""
    PERFORMANCE = "performance"
    RISK = "risk"
    CONSISTENCY = "consistency"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    COMPLIANCE = "compliance"


@dataclass
class ValidationCriterion:
    """Individual validation criterion"""
    
    name: str
    description: str
    validation_type: ValidationType
    level: ValidationLevel
    
    # Validation parameters
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    tolerance: Optional[float] = None
    
    # Weight and importance
    weight: float = 1.0
    required: bool = True
    
    # Validation function
    validation_function: Optional[str] = None
    
    def validate(self, value: float) -> Dict[str, Any]:
        """Validate a value against this criterion"""
        
        result = {
            'criterion_name': self.name,
            'passed': True,
            'value': value,
            'message': '',
            'severity': 'info'
        }
        
        try:
            # Check minimum value
            if self.min_value is not None and value < self.min_value:
                result['passed'] = False
                result['message'] = f"Value {value:.4f} is below minimum {self.min_value:.4f}"
                result['severity'] = 'error' if self.required else 'warning'
            
            # Check maximum value
            if self.max_value is not None and value > self.max_value:
                result['passed'] = False
                result['message'] = f"Value {value:.4f} is above maximum {self.max_value:.4f}"
                result['severity'] = 'error' if self.required else 'warning'
            
            # Check target value with tolerance
            if self.target_value is not None and self.tolerance is not None:
                if abs(value - self.target_value) > self.tolerance:
                    result['passed'] = False
                    result['message'] = f"Value {value:.4f} is outside target range {self.target_value:.4f} ± {self.tolerance:.4f}"
                    result['severity'] = 'warning'
            
            # Custom validation function
            if self.validation_function:
                # This would be implemented with actual validation logic
                pass
            
            if result['passed']:
                result['message'] = f"Value {value:.4f} meets criterion {self.name}"
            
        except Exception as e:
            result['passed'] = False
            result['message'] = f"Validation error: {str(e)}"
            result['severity'] = 'error'
        
        return result


@dataclass
class ValidationCriteriaSet:
    """Set of validation criteria for a specific purpose"""
    
    name: str
    description: str
    criteria: List[ValidationCriterion]
    level: ValidationLevel = ValidationLevel.MODERATE
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against all criteria in this set"""
        
        results = {
            'criteria_set': self.name,
            'overall_passed': True,
            'total_criteria': len(self.criteria),
            'passed_criteria': 0,
            'failed_criteria': 0,
            'warnings': 0,
            'errors': 0,
            'results': []
        }
        
        for criterion in self.criteria:
            try:
                # Get value for this criterion
                value = self._get_value_for_criterion(data, criterion)
                
                if value is not None:
                    # Validate the value
                    criterion_result = criterion.validate(value)
                    results['results'].append(criterion_result)
                    
                    # Update counters
                    if criterion_result['passed']:
                        results['passed_criteria'] += 1
                    else:
                        results['failed_criteria'] += 1
                        if criterion.required:
                            results['overall_passed'] = False
                    
                    # Count warnings and errors
                    if criterion_result['severity'] == 'warning':
                        results['warnings'] += 1
                    elif criterion_result['severity'] == 'error':
                        results['errors'] += 1
                
                else:
                    # Missing value
                    missing_result = {
                        'criterion_name': criterion.name,
                        'passed': False,
                        'value': None,
                        'message': f"Required value for criterion {criterion.name} is missing",
                        'severity': 'error' if criterion.required else 'warning'
                    }
                    results['results'].append(missing_result)
                    results['failed_criteria'] += 1
                    
                    if criterion.required:
                        results['overall_passed'] = False
                        results['errors'] += 1
                    else:
                        results['warnings'] += 1
            
            except Exception as e:
                logger.error(f"Error validating criterion {criterion.name}: {e}")
                error_result = {
                    'criterion_name': criterion.name,
                    'passed': False,
                    'value': None,
                    'message': f"Validation error: {str(e)}",
                    'severity': 'error'
                }
                results['results'].append(error_result)
                results['failed_criteria'] += 1
                results['errors'] += 1
                results['overall_passed'] = False
        
        return results
    
    def _get_value_for_criterion(self, data: Dict[str, Any], criterion: ValidationCriterion) -> Optional[float]:
        """Get the value for a specific criterion from the data"""
        
        # Map criterion names to data fields
        field_mapping = {
            'total_trades': 'total_trades',
            'win_rate': 'win_rate',
            'profit_factor': 'profit_factor',
            'max_drawdown': 'max_drawdown',
            'sharpe_ratio': 'sharpe_ratio',
            'sortino_ratio': 'sortino_ratio',
            'calmar_ratio': 'calmar_ratio',
            'var_95': 'var_95',
            'cvar_95': 'cvar_95',
            'avg_trade_duration': 'avg_trade_duration',
            'consistency_score': 'consistency_score',
            'robustness_score': 'robustness_score',
            'efficiency_score': 'efficiency_score',
            'stability_score': 'stability_score',
            'reliability_score': 'reliability_score'
        }
        
        field_name = field_mapping.get(criterion.name)
        if field_name and field_name in data:
            return float(data[field_name])
        
        return None


class ValidationCriteriaFactory:
    """Factory for creating validation criteria sets"""
    
    @staticmethod
    def create_performance_criteria(level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationCriteriaSet:
        """Create performance validation criteria"""
        
        criteria = [
            ValidationCriterion(
                name='total_trades',
                description='Minimum number of trades',
                validation_type=ValidationType.PERFORMANCE,
                level=level,
                min_value=10.0,
                required=True
            ),
            ValidationCriterion(
                name='win_rate',
                description='Minimum win rate',
                validation_type=ValidationType.PERFORMANCE,
                level=level,
                min_value=0.3,
                required=True
            ),
            ValidationCriterion(
                name='profit_factor',
                description='Minimum profit factor',
                validation_type=ValidationType.PERFORMANCE,
                level=level,
                min_value=1.1,
                required=True
            )
        ]
        
        return ValidationCriteriaSet(
            name='performance_criteria',
            description='Performance validation criteria',
            criteria=criteria,
            level=level
        )
    
    @staticmethod
    def create_risk_criteria(level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationCriteriaSet:
        """Create risk validation criteria"""
        
        criteria = [
            ValidationCriterion(
                name='max_drawdown',
                description='Maximum drawdown',
                validation_type=ValidationType.RISK,
                level=level,
                max_value=0.2,
                required=True
            ),
            ValidationCriterion(
                name='var_95',
                description='Value at Risk (95%)',
                validation_type=ValidationType.RISK,
                level=level,
                max_value=0.05,
                required=True
            ),
            ValidationCriterion(
                name='cvar_95',
                description='Conditional Value at Risk (95%)',
                validation_type=ValidationType.RISK,
                level=level,
                max_value=0.06,
                required=True
            )
        ]
        
        return ValidationCriteriaSet(
            name='risk_criteria',
            description='Risk validation criteria',
            criteria=criteria,
            level=level
        )
    
    @staticmethod
    def create_consistency_criteria(level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationCriteriaSet:
        """Create consistency validation criteria"""
        
        criteria = [
            ValidationCriterion(
                name='consistency_score',
                description='Minimum consistency score',
                validation_type=ValidationType.CONSISTENCY,
                level=level,
                min_value=0.3,
                required=True
            ),
            ValidationCriterion(
                name='stability_score',
                description='Minimum stability score',
                validation_type=ValidationType.CONSISTENCY,
                level=level,
                min_value=0.3,
                required=True
            ),
            ValidationCriterion(
                name='reliability_score',
                description='Minimum reliability score',
                validation_type=ValidationType.CONSISTENCY,
                level=level,
                min_value=0.3,
                required=True
            )
        ]
        
        return ValidationCriteriaSet(
            name='consistency_criteria',
            description='Consistency validation criteria',
            criteria=criteria,
            level=level
        )
    
    @staticmethod
    def create_efficiency_criteria(level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationCriteriaSet:
        """Create efficiency validation criteria"""
        
        criteria = [
            ValidationCriterion(
                name='efficiency_score',
                description='Minimum efficiency score',
                validation_type=ValidationType.EFFICIENCY,
                level=level,
                min_value=0.3,
                required=True
            ),
            ValidationCriterion(
                name='sharpe_ratio',
                description='Minimum Sharpe ratio',
                validation_type=ValidationType.EFFICIENCY,
                level=level,
                min_value=0.5,
                required=True
            ),
            ValidationCriterion(
                name='calmar_ratio',
                description='Minimum Calmar ratio',
                validation_type=ValidationType.EFFICIENCY,
                level=level,
                min_value=0.5,
                required=True
            )
        ]
        
        return ValidationCriteriaSet(
            name='efficiency_criteria',
            description='Efficiency validation criteria',
            criteria=criteria,
            level=level
        )
    
    @staticmethod
    def create_robustness_criteria(level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationCriteriaSet:
        """Create robustness validation criteria"""
        
        criteria = [
            ValidationCriterion(
                name='robustness_score',
                description='Minimum robustness score',
                validation_type=ValidationType.ROBUSTNESS,
                level=level,
                min_value=0.3,
                required=True
            ),
            ValidationCriterion(
                name='avg_trade_duration',
                description='Average trade duration',
                validation_type=ValidationType.ROBUSTNESS,
                level=level,
                min_value=1.0,
                max_value=100.0,
                required=True
            )
        ]
        
        return ValidationCriteriaSet(
            name='robustness_criteria',
            description='Robustness validation criteria',
            criteria=criteria,
            level=level
        )
    
    @staticmethod
    def create_comprehensive_criteria(level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationCriteriaSet:
        """Create comprehensive validation criteria combining all types"""
        
        criteria = []
        
        # Add all criteria from different types
        criteria.extend(ValidationCriteriaFactory.create_performance_criteria(level).criteria)
        criteria.extend(ValidationCriteriaFactory.create_risk_criteria(level).criteria)
        criteria.extend(ValidationCriteriaFactory.create_consistency_criteria(level).criteria)
        criteria.extend(ValidationCriteriaFactory.create_efficiency_criteria(level).criteria)
        criteria.extend(ValidationCriteriaFactory.create_robustness_criteria(level).criteria)
        
        return ValidationCriteriaSet(
            name='comprehensive_criteria',
            description='Comprehensive validation criteria',
            criteria=criteria,
            level=level
        )
    
    @staticmethod
    def create_custom_criteria(
        name: str,
        description: str,
        criteria_definitions: List[Dict[str, Any]],
        level: ValidationLevel = ValidationLevel.MODERATE
    ) -> ValidationCriteriaSet:
        """Create custom validation criteria from definitions"""
        
        criteria = []
        
        for defn in criteria_definitions:
            criterion = ValidationCriterion(
                name=defn['name'],
                description=defn['description'],
                validation_type=ValidationType(defn['validation_type']),
                level=level,
                min_value=defn.get('min_value'),
                max_value=defn.get('max_value'),
                target_value=defn.get('target_value'),
                tolerance=defn.get('tolerance'),
                weight=defn.get('weight', 1.0),
                required=defn.get('required', True),
                validation_function=defn.get('validation_function')
            )
            criteria.append(criterion)
        
        return ValidationCriteriaSet(
            name=name,
            description=description,
            criteria=criteria,
            level=level
        )

