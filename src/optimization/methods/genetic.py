"""
Genetic algorithm optimization method.
"""
import logging
import random
from typing import List, Any, Dict

from src.backtesting.config import BacktestConfig
from src.optimization.results import OptimizationResult, OptimizationResults

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    """Genetic algorithm optimizer."""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    async def optimize(
        self,
        backtest_config: BacktestConfig,
        population_size: int = 50,
        generations: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7
    ) -> OptimizationResults:
        """Run genetic algorithm."""
        logger.info(f"Genetic algorithm: {generations} generations, pop size {population_size}")
        
        # 1. Initialize Population
        population = [
            self.optimizer.parameter_space.sample_random()
            for _ in range(population_size)
        ]
        
        all_results = []
        best_fitness_history = []
        
        for gen in range(generations):
            logger.info(f"Generation {gen+1}/{generations}")
            
            # 2. Evaluate Fitness
            fitness_scores = []
            gen_results = []
            
            for params in population:
                backtest_result = await self.optimizer.run_backtest(backtest_config, params)
                objective_value = self.optimizer.get_objective_value(backtest_result)
                
                fitness_scores.append(objective_value)
                result = OptimizationResult(
                    params=params,
                    objective_value=objective_value,
                    backtest_result=backtest_result
                )
                gen_results.append(result)
                all_results.append(result)
            
            current_best = max(fitness_scores) if fitness_scores else 0
            best_fitness_history.append(current_best)
            logger.info(f"Generation {gen+1} best fitness: {current_best:.4f}")
            
            # If last generation, break
            if gen == generations - 1:
                break
                
            # 3. Selection (Tournament)
            selected = self._tournament_selection(population, fitness_scores)
            
            # 4. Crossover
            offspring = self._crossover(selected, crossover_rate)
            
            # 5. Mutation
            offspring = self._mutate(offspring, mutation_rate)
            
            # Elitism: Keep best individual
            best_idx = fitness_scores.index(max(fitness_scores))
            offspring[0] = population[best_idx]
            
            # Replace population
            population = offspring
            
        sorted_results = sorted(all_results, key=lambda r: r.objective_value, reverse=True)
        
        return OptimizationResults(
            method="genetic",
            objective=self.optimizer.objective,
            results=sorted_results,
            best_params=sorted_results[0].params if sorted_results else {},
            convergence_curve=best_fitness_history
        )
    
    def _tournament_selection(self, population, fitness, tournament_size=3):
        """Select parents using tournament selection."""
        selected = []
        pop_size = len(population)
        
        for _ in range(pop_size):
            candidates_indices = random.sample(range(pop_size), tournament_size)
            best_idx = max(candidates_indices, key=lambda i: fitness[i])
            selected.append(population[best_idx])
            
        return selected

    def _crossover(self, parents, crossover_rate):
        """Perform crossover to create offspring."""
        offspring = []
        num_parents = len(parents)
        
        for i in range(0, num_parents, 2):
            p1 = parents[i]
            # Handle odd number of parents
            p2 = parents[min(i+1, num_parents-1)]
            
            if random.random() < crossover_rate:
                # Uniform crossover
                child1 = {}
                child2 = {}
                
                for key in p1.keys():
                    if random.random() < 0.5:
                        child1[key] = p1[key]
                        child2[key] = p2[key]
                    else:
                        child1[key] = p2[key]
                        child2[key] = p1[key]
                
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(p1.copy())
                offspring.append(p2.copy())
                
        # Truncate if we generated too many
        return offspring[:num_parents]

    def _mutate(self, population, mutation_rate):
        """Mutate population."""
        mutated_pop = []
        parameter_space = self.optimizer.parameter_space
        
        for individual in population:
            mutated = individual.copy()
            
            for key in mutated.keys():
                if random.random() < mutation_rate:
                    # Replace with random value from parameter definition
                    param_def = parameter_space.get_parameter(key)
                    mutated[key] = param_def.sample_random()
            
            mutated_pop.append(mutated)
            
        return mutated_pop
