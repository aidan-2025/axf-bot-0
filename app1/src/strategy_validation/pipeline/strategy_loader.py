#!/usr/bin/env python3
"""
Strategy Loader

Loads and validates strategy definitions for backtesting.
"""

import json
import importlib
import inspect
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Type, Union
from datetime import datetime
import logging
import os
import sys

logger = logging.getLogger(__name__)


@dataclass
class StrategyDefinition:
    """Strategy definition data structure"""
    
    # Basic info
    strategy_id: str
    strategy_name: str
    strategy_type: str
    description: str
    
    # Strategy class info
    class_name: str
    module_path: str
    
    # Parameters
    parameters: Dict[str, Any]
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    version: str = "1.0.0"
    
    # Validation info
    is_valid: bool = True
    validation_errors: List[str] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'strategy_type': self.strategy_type,
            'description': self.description,
            'class_name': self.class_name,
            'module_path': self.module_path,
            'parameters': self.parameters,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'is_valid': self.is_valid,
            'validation_errors': self.validation_errors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyDefinition':
        """Create from dictionary"""
        return cls(
            strategy_id=data['strategy_id'],
            strategy_name=data['strategy_name'],
            strategy_type=data['strategy_type'],
            description=data['description'],
            class_name=data['class_name'],
            module_path=data['module_path'],
            parameters=data['parameters'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            version=data.get('version', '1.0.0'),
            is_valid=data.get('is_valid', True),
            validation_errors=data.get('validation_errors', [])
        )


class StrategyLoader:
    """Loads and validates strategy definitions"""
    
    def __init__(self, strategy_directories: List[str] = None):
        self.strategy_directories = strategy_directories or ['src/strategy_generation/templates']
        self.logger = logging.getLogger(__name__)
        
        # Add strategy directories to Python path
        for directory in self.strategy_directories:
            if directory not in sys.path:
                sys.path.append(directory)
        
        self.logger.info(f"StrategyLoader initialized with directories: {self.strategy_directories}")
    
    def load_strategy_from_json(self, json_file_path: str) -> StrategyDefinition:
        """Load strategy definition from JSON file"""
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            return self._create_strategy_definition_from_data(data)
            
        except Exception as e:
            self.logger.error(f"Failed to load strategy from JSON {json_file_path}: {e}")
            raise
    
    def load_strategy_from_dict(self, data: Dict[str, Any]) -> StrategyDefinition:
        """Load strategy definition from dictionary"""
        try:
            return self._create_strategy_definition_from_data(data)
        except Exception as e:
            self.logger.error(f"Failed to load strategy from dictionary: {e}")
            raise
    
    def load_strategy_from_class(self, strategy_class: Type, 
                                strategy_id: str, 
                                parameters: Dict[str, Any] = None) -> StrategyDefinition:
        """Load strategy definition from Python class"""
        try:
            # Get class metadata
            class_name = strategy_class.__name__
            module_path = strategy_class.__module__
            
            # Extract parameters from class if not provided
            if parameters is None:
                parameters = self._extract_parameters_from_class(strategy_class)
            
            # Create strategy definition
            definition = StrategyDefinition(
                strategy_id=strategy_id,
                strategy_name=getattr(strategy_class, 'name', class_name),
                strategy_type=getattr(strategy_class, 'strategy_type', 'custom'),
                description=getattr(strategy_class, 'description', f"Strategy {class_name}"),
                class_name=class_name,
                module_path=module_path,
                parameters=parameters or {},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Validate the definition
            self._validate_strategy_definition(definition)
            
            return definition
            
        except Exception as e:
            self.logger.error(f"Failed to load strategy from class {strategy_class}: {e}")
            raise
    
    def load_strategies_from_directory(self, directory: str) -> List[StrategyDefinition]:
        """Load all strategies from a directory"""
        strategies = []
        
        try:
            if not os.path.exists(directory):
                self.logger.warning(f"Strategy directory does not exist: {directory}")
                return strategies
            
            # Look for Python files
            for filename in os.listdir(directory):
                if filename.endswith('.py') and not filename.startswith('__'):
                    module_name = filename[:-3]
                    try:
                        # Import the module
                        module = importlib.import_module(module_name)
                        
                        # Find strategy classes
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                hasattr(obj, '__bases__') and 
                                'StrategyTemplate' in [base.__name__ for base in obj.__bases__]):
                                
                                strategy_def = self.load_strategy_from_class(
                                    obj, 
                                    f"{module_name}_{name}",
                                    {}
                                )
                                strategies.append(strategy_def)
                                
                    except Exception as e:
                        self.logger.warning(f"Failed to load strategy from {filename}: {e}")
                        continue
            
            self.logger.info(f"Loaded {len(strategies)} strategies from {directory}")
            return strategies
            
        except Exception as e:
            self.logger.error(f"Failed to load strategies from directory {directory}: {e}")
            return strategies
    
    def load_strategies_from_json_batch(self, json_files: List[str]) -> List[StrategyDefinition]:
        """Load multiple strategies from JSON files"""
        strategies = []
        
        for json_file in json_files:
            try:
                strategy = self.load_strategy_from_json(json_file)
                strategies.append(strategy)
            except Exception as e:
                self.logger.error(f"Failed to load strategy from {json_file}: {e}")
                continue
        
        self.logger.info(f"Loaded {len(strategies)} strategies from {len(json_files)} JSON files")
        return strategies
    
    def instantiate_strategy_class(self, definition: StrategyDefinition) -> Type:
        """Instantiate strategy class from definition"""
        try:
            # Import the module
            module = importlib.import_module(definition.module_path)
            
            # Get the class
            strategy_class = getattr(module, definition.class_name)
            
            # Validate it's a proper strategy class
            if not self._is_valid_strategy_class(strategy_class):
                raise ValueError(f"Class {definition.class_name} is not a valid strategy class")
            
            return strategy_class
            
        except Exception as e:
            self.logger.error(f"Failed to instantiate strategy class {definition.class_name}: {e}")
            raise
    
    def _create_strategy_definition_from_data(self, data: Dict[str, Any]) -> StrategyDefinition:
        """Create strategy definition from data dictionary"""
        # Parse timestamps
        created_at = datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
        updated_at = datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat()))
        
        definition = StrategyDefinition(
            strategy_id=data['strategy_id'],
            strategy_name=data['strategy_name'],
            strategy_type=data['strategy_type'],
            description=data.get('description', ''),
            class_name=data['class_name'],
            module_path=data['module_path'],
            parameters=data.get('parameters', {}),
            created_at=created_at,
            updated_at=updated_at,
            version=data.get('version', '1.0.0'),
            is_valid=data.get('is_valid', True),
            validation_errors=data.get('validation_errors', [])
        )
        
        # Validate the definition
        self._validate_strategy_definition(definition)
        
        return definition
    
    def _extract_parameters_from_class(self, strategy_class: Type) -> Dict[str, Any]:
        """Extract parameters from strategy class"""
        parameters = {}
        
        try:
            # Look for params attribute (Backtrader style)
            if hasattr(strategy_class, 'params'):
                params = strategy_class.params
                if hasattr(params, '__dict__'):
                    parameters.update(params.__dict__)
                elif isinstance(params, dict):
                    parameters.update(params)
            
            # Look for __init__ parameters
            init_signature = inspect.signature(strategy_class.__init__)
            for param_name, param in init_signature.parameters.items():
                if param_name != 'self' and param.default != inspect.Parameter.empty:
                    parameters[param_name] = param.default
            
        except Exception as e:
            self.logger.warning(f"Failed to extract parameters from class {strategy_class}: {e}")
        
        return parameters
    
    def _validate_strategy_definition(self, definition: StrategyDefinition) -> None:
        """Validate strategy definition"""
        errors = []
        
        # Check required fields
        if not definition.strategy_id:
            errors.append("Strategy ID is required")
        
        if not definition.strategy_name:
            errors.append("Strategy name is required")
        
        if not definition.class_name:
            errors.append("Class name is required")
        
        if not definition.module_path:
            errors.append("Module path is required")
        
        # Check if class can be imported
        try:
            strategy_class = self.instantiate_strategy_class(definition)
            if not self._is_valid_strategy_class(strategy_class):
                errors.append(f"Class {definition.class_name} is not a valid strategy class")
        except Exception as e:
            errors.append(f"Cannot import strategy class: {e}")
        
        # Update validation status
        definition.is_valid = len(errors) == 0
        definition.validation_errors = errors
        
        if errors:
            self.logger.warning(f"Strategy {definition.strategy_id} validation failed: {errors}")
    
    def _is_valid_strategy_class(self, strategy_class: Type) -> bool:
        """Check if class is a valid strategy class"""
        try:
            # Check if it has required methods
            required_methods = ['next', 'initialize']
            for method in required_methods:
                if not hasattr(strategy_class, method):
                    return False
            
            # Check if it's a Backtrader strategy or our custom strategy
            import backtrader as bt
            if not (issubclass(strategy_class, bt.Strategy) or 
                   hasattr(strategy_class, '__bases__') and 
                   'StrategyTemplate' in [base.__name__ for base in strategy_class.__bases__]):
                return False
            
            return True
            
        except Exception:
            return False
    
    def save_strategy_definition(self, definition: StrategyDefinition, 
                               output_path: str) -> None:
        """Save strategy definition to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(definition.to_dict(), f, indent=2)
            
            self.logger.info(f"Saved strategy definition to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save strategy definition to {output_path}: {e}")
            raise
    
    def get_strategy_summary(self, definitions: List[StrategyDefinition]) -> Dict[str, Any]:
        """Get summary of loaded strategies"""
        total = len(definitions)
        valid = len([d for d in definitions if d.is_valid])
        invalid = total - valid
        
        strategy_types = {}
        for definition in definitions:
            strategy_type = definition.strategy_type
            strategy_types[strategy_type] = strategy_types.get(strategy_type, 0) + 1
        
        return {
            'total_strategies': total,
            'valid_strategies': valid,
            'invalid_strategies': invalid,
            'validation_rate': valid / total if total > 0 else 0,
            'strategy_types': strategy_types,
            'strategy_ids': [d.strategy_id for d in definitions]
        }

