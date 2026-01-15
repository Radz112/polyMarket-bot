"""
Visualization for optimization results.
"""
from typing import Optional
from src.optimization.results import OptimizationResults

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def plot_convergence(results: OptimizationResults, save_path: Optional[str] = None):
    """Plot optimization convergence."""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib/Pandas not available for plotting")
        return
        
    if not results.convergence_curve:
        print("No convergence data available")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(results.convergence_curve)
    plt.title(f"Optimization Convergence ({results.method})")
    plt.xlabel("Iteration")
    plt.ylabel(f"Objective ({results.objective})")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_parameter_sensitivity(
    results: OptimizationResults, 
    param_name: str, 
    save_path: Optional[str] = None
):
    """Plot parameter sensitivity."""
    if not PLOTTING_AVAILABLE:
        return
        
    df = results.get_parameter_sensitivity(param_name)
    if df.empty:
        return
        
    plt.figure(figsize=(10, 6))
    
    # Plot mean with error bars (std dev)
    plt.errorbar(
        df.index, 
        df['mean'], 
        yerr=df['std'], 
        fmt='o-', 
        capsize=5
    )
    
    plt.title(f"Sensitivity: {param_name}")
    plt.xlabel(param_name)
    plt.ylabel(f"Objective ({results.objective})")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
