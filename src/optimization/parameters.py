"""
Parameter definition and management.
"""
import random
import itertools
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union
import numpy as np


@dataclass
class Parameter:
    """Definition of a single parameter to optimize."""
    name: str
    param_type: str  # "float", "int", "categorical"
    
    # For numeric parameters
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    
    # For categorical parameters
    choices: Optional[List[Any]] = None
    
    # Current value
    value: Any = None
    
    def sample_random(self) -> Any:
        """Sample random value from parameter space."""
        if self.param_type == "float":
            if self.step:
                # Sample from stepped range
                steps = int((self.max_value - self.min_value) / self.step)
                return self.min_value + (random.randint(0, steps) * self.step)
            return random.uniform(self.min_value, self.max_value)
            
        elif self.param_type == "int":
            step = int(self.step) if self.step else 1
            return random.randrange(int(self.min_value), int(self.max_value) + 1, step)
            
        elif self.param_type == "categorical":
            return random.choice(self.choices)
    
    def get_grid(self) -> List[Any]:
        """Get all values for grid search."""
        if self.param_type == "float":
            # Use numpy for float ranges to avoid precision issues
            return [float(x) for x in np.arange(self.min_value, self.max_value + (self.step/1000), self.step)]
        elif self.param_type == "int":
            step = int(self.step) if self.step else 1
            return list(range(int(self.min_value), int(self.max_value) + 1, step))
        elif self.param_type == "categorical":
            return self.choices


class ParameterSpace:
    """Manages a collection of parameters."""
    
    def __init__(self, parameters: List[Parameter]):
        self.parameters = {p.name: p for p in parameters}
    
    def get_parameter(self, name: str) -> Parameter:
        """Get parameter by name."""
        return self.parameters[name]
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameter values."""
        return {name: p.value for name, p in self.parameters.items()}
    
    def sample_random(self) -> Dict[str, Any]:
        """Sample random parameter combination."""
        return {name: p.sample_random() for name, p in self.parameters.items()}
    
    def get_grid_combinations(self) -> List[Dict[str, Any]]:
        """Get all combinations for grid search."""
        grids = {name: p.get_grid() for name, p in self.parameters.items()}
        keys = list(grids.keys())
        values = list(grids.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        return combinations
    
    def total_combinations(self) -> int:
        """Total number of grid combinations."""
        total = 1
        for p in self.parameters.values():
            total *= len(p.get_grid())
        return total
    
    def params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert params dict to numerical array (for Bayesian opt)."""
        values = []
        for name, p in self.parameters.items():
            val = params[name]
            if p.param_type == "categorical":
                # Simple encoding: index in choices
                values.append(p.choices.index(val))
            else:
                values.append(val)
        return np.array(values)
    
    def array_to_params(self, array: np.ndarray) -> Dict[str, Any]:
        """Convert numerical array back to params dict."""
        params = {}
        for i, (name, p) in enumerate(self.parameters.items()):
            val = array[i]
            if p.param_type == "categorical":
                idx = int(round(val))
                idx = max(0, min(idx, len(p.choices) - 1))
                params[name] = p.choices[idx]
            elif p.param_type == "int":
                params[name] = int(round(val))
            else:
                params[name] = float(val)
        return params
