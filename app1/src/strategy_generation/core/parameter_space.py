"""
Parameter space definitions for strategy optimization
"""

from enum import Enum
from typing import Dict, Any, List, Union, Tuple
from dataclasses import dataclass
import numpy as np


class ParameterType(Enum):
    """Parameter type enumeration"""
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    RANGE = "range"


@dataclass
class ParameterDefinition:
    """Definition of a single parameter"""
    name: str
    param_type: ParameterType
    min_value: Union[int, float, None] = None
    max_value: Union[int, float, None] = None
    default_value: Union[int, float, bool, str, None] = None
    categories: List[Union[str, int, float]] = None
    step: Union[int, float, None] = None
    description: str = ""
    constraints: List[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []
        if self.constraints is None:
            self.constraints = []
    
    def validate_value(self, value: Any) -> bool:
        """Validate if a value is within parameter constraints"""
        if self.param_type == ParameterType.INTEGER:
            if not isinstance(value, (int, np.integer)):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
            return True
            
        elif self.param_type == ParameterType.FLOAT:
            if not isinstance(value, (float, int, np.floating, np.integer)):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
            return True
            
        elif self.param_type == ParameterType.BOOLEAN:
            return isinstance(value, bool)
            
        elif self.param_type == ParameterType.CATEGORICAL:
            return value in self.categories
            
        elif self.param_type == ParameterType.RANGE:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return False
            min_val, max_val = value
            if self.min_value is not None and min_val < self.min_value:
                return False
            if self.max_value is not None and max_val > self.max_value:
                return False
            return min_val <= max_val
            
        return False
    
    def normalize_value(self, value: Any) -> float:
        """Normalize value to [0, 1] range for genetic algorithm"""
        if self.param_type == ParameterType.INTEGER:
            if self.min_value is None or self.max_value is None:
                return 0.5
            return (value - self.min_value) / (self.max_value - self.min_value)
            
        elif self.param_type == ParameterType.FLOAT:
            if self.min_value is None or self.max_value is None:
                return 0.5
            return (value - self.min_value) / (self.max_value - self.min_value)
            
        elif self.param_type == ParameterType.BOOLEAN:
            return 1.0 if value else 0.0
            
        elif self.param_type == ParameterType.CATEGORICAL:
            if not self.categories:
                return 0.5
            try:
                index = self.categories.index(value)
                return index / (len(self.categories) - 1)
            except ValueError:
                return 0.5
                
        elif self.param_type == ParameterType.RANGE:
            # Normalize range as average of min and max
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return 0.5
            min_val, max_val = value
            avg_val = (min_val + max_val) / 2
            if self.min_value is None or self.max_value is None:
                return 0.5
            return (avg_val - self.min_value) / (self.max_value - self.min_value)
            
        return 0.5
    
    def denormalize_value(self, normalized_value: float) -> Any:
        """Convert normalized value back to original parameter value"""
        if self.param_type == ParameterType.INTEGER:
            if self.min_value is None or self.max_value is None:
                return int(normalized_value * 100)
            return int(self.min_value + normalized_value * (self.max_value - self.min_value))
            
        elif self.param_type == ParameterType.FLOAT:
            if self.min_value is None or self.max_value is None:
                return float(normalized_value)
            return self.min_value + normalized_value * (self.max_value - self.min_value)
            
        elif self.param_type == ParameterType.BOOLEAN:
            return normalized_value > 0.5
            
        elif self.param_type == ParameterType.CATEGORICAL:
            if not self.categories:
                return self.categories[0] if self.categories else None
            index = int(normalized_value * (len(self.categories) - 1))
            index = max(0, min(index, len(self.categories) - 1))
            return self.categories[index]
            
        elif self.param_type == ParameterType.RANGE:
            if self.min_value is None or self.max_value is None:
                return [0.0, 1.0]
            center = self.min_value + normalized_value * (self.max_value - self.min_value)
            range_size = (self.max_value - self.min_value) * 0.1  # 10% of total range
            return [center - range_size/2, center + range_size/2]
            
        return None


class ParameterSpace:
    """
    Manages parameter space definitions for strategy optimization
    """
    
    def __init__(self):
        self.parameters: Dict[str, ParameterDefinition] = {}
        self.constraints: List[str] = []
    
    def add_parameter(self, param_def: ParameterDefinition):
        """Add a parameter definition"""
        self.parameters[param_def.name] = param_def
    
    def add_constraint(self, constraint: str):
        """Add a parameter constraint"""
        self.constraints.append(constraint)
    
    def get_parameter(self, name: str) -> ParameterDefinition:
        """Get parameter definition by name"""
        return self.parameters.get(name)
    
    def get_all_parameters(self) -> Dict[str, ParameterDefinition]:
        """Get all parameter definitions"""
        return self.parameters.copy()
    
    def validate_parameters(self, param_values: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameter values against definitions
        
        Args:
            param_values: Dictionary of parameter values
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if all required parameters are present
        for name, param_def in self.parameters.items():
            if name not in param_values:
                errors.append(f"Missing required parameter: {name}")
                continue
                
            if not param_def.validate_value(param_values[name]):
                errors.append(f"Invalid value for parameter {name}: {param_values[name]}")
        
        # Check for extra parameters
        for name in param_values:
            if name not in self.parameters:
                errors.append(f"Unknown parameter: {name}")
        
        # Validate constraints
        for constraint in self.constraints:
            if not self._evaluate_constraint(constraint, param_values):
                errors.append(f"Constraint violation: {constraint}")
        
        return len(errors) == 0, errors
    
    def _evaluate_constraint(self, constraint: str, param_values: Dict[str, Any]) -> bool:
        """Evaluate a constraint expression"""
        try:
            # Simple constraint evaluation - can be extended for more complex expressions
            # For now, just return True - implement proper constraint evaluation as needed
            return True
        except Exception:
            return False
    
    def get_parameter_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization algorithms"""
        bounds = []
        for param_def in self.parameters.values():
            if param_def.param_type in [ParameterType.INTEGER, ParameterType.FLOAT]:
                min_val = param_def.min_value if param_def.min_value is not None else 0.0
                max_val = param_def.max_value if param_def.max_value is not None else 1.0
                bounds.append((min_val, max_val))
            else:
                bounds.append((0.0, 1.0))  # Normalized bounds
        return bounds
    
    def encode_parameters(self, param_values: Dict[str, Any]) -> List[float]:
        """Encode parameters to normalized vector for genetic algorithm"""
        encoded = []
        for name, param_def in self.parameters.items():
            if name in param_values:
                encoded.append(param_def.normalize_value(param_values[name]))
            else:
                encoded.append(0.5)  # Default normalized value
        return encoded
    
    def decode_parameters(self, encoded_values: List[float]) -> Dict[str, Any]:
        """Decode normalized vector back to parameter values"""
        decoded = {}
        param_names = list(self.parameters.keys())
        
        for i, encoded_value in enumerate(encoded_values):
            if i < len(param_names):
                param_name = param_names[i]
                param_def = self.parameters[param_name]
                decoded[param_name] = param_def.denormalize_value(encoded_value)
        
        return decoded
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters"""
        return len(self.parameters)
    
    def get_categorical_parameters(self) -> List[str]:
        """Get list of categorical parameter names"""
        return [name for name, param_def in self.parameters.items() 
                if param_def.param_type == ParameterType.CATEGORICAL]
    
    def get_numerical_parameters(self) -> List[str]:
        """Get list of numerical parameter names"""
        return [name for name, param_def in self.parameters.items() 
                if param_def.param_type in [ParameterType.INTEGER, ParameterType.FLOAT]]
    
    def __str__(self) -> str:
        return f"ParameterSpace({len(self.parameters)} parameters)"
    
    def __repr__(self) -> str:
        return f"ParameterSpace(parameters={list(self.parameters.keys())})"

