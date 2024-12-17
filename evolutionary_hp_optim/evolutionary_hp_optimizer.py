from typing import Optional, Tuple, List, Dict, Any
import abc
import random
import logging as lg
from decimal import Decimal, ROUND_HALF_UP
from time import perf_counter
import numpy as np
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm


class BaseEvolutionaryOptimizer:

    @staticmethod
    def _set_random_state(random_state: int):
        """
        Set deterministic seed.
        :param random_state: seed or random_state for reproducibility.
        """
        random.seed(random_state)
        np.random.seed(random_state)

    @staticmethod
    def _get_param_grid(paramdict: Dict[str, List], k: int) -> List[Dict[str, Any]]:
        """
        Obtains k random individuals from all posible combinations in a parameter dict
        :param paramdict: Parameter dict
        :param k: Number of random individuals
        """
        return [{key: random.choice(paramdict[key]) for key in paramdict} for _ in range(k)]

    def initialize_population(
            self, population_size: int,
            individuals_paramdict: Dict[type, Dict[str, List]],
            population: Optional[Dict] = None,
            individual_params: Optional[Dict] = None,
            families: Optional[Dict] = None) -> Tuple[Dict[int, Any],  Dict[int, Any], Dict[int, str]]:
        """
        Initializes population.
        :param population_size: Num individuals
        :param individuals_paramdict: Tuple of dicts with param combinations used for evolution.
        :param population: current population
        :param individual_params: current individual parameters
        :param families: current families
        :return: Population and families
        """
        assert type(population) == type(families), f'If population is not None families cant be None and viceversa.'
        population = {} if population is None else population
        families = {} if families is None else families
        individual_params = {} if individual_params is None else individual_params
        family_fraction = (population_size - len(population)) // len(list(individuals_paramdict))
        idx = 0 if len(population) == 0 else max(list(population))
        while len(population) < population_size:
            family = random.choice(list(individuals_paramdict))
            try:
                param_grid = self._get_param_grid(individuals_paramdict[family], k=family_fraction)
                population.update({
                    idx + 1: family(**pg) for pg in param_grid
                })
                individual_params.update({
                    idx + 1: pg for pg in param_grid
                })
                families.update({
                    idx + 1: family.__name__
                })
                idx += 1
            except ValueError:
                pass
        return population, individual_params, families

    @staticmethod
    def mutate_param(param: Optional[Any], mutation_probability: float, param_spec: List[Any]):
        """
        Mutates single param value
        :param param: Current parameter
        :param mutation_probability: Mutation probability
        :param param_spec: Posible parameter's values
        :return: New parameter value
        """
        if random.random() < mutation_probability and param is not None and len(param_spec) > 1:
            if any(isinstance(p, (str, bool)) for p in param_spec):
                return random.choice(list(set(param_spec).difference([param])))
            elif any(isinstance(p, float) for p in param_spec):
                return min(max(param_spec),
                           max(min(param_spec),
                               param + ((-1) ** (random.random() >= 0.5)) * random.random() * 0.5)
                           )
            elif any(isinstance(p, int) for p in param_spec):
                return int(
                    min(max(param_spec),
                        max(min(param_spec),
                            param * (2 * random.random())
                            )
                        )
                )
        return param

    def mutate_inidividual(self,
                           individual: Any,
                           individual_params: Dict[str, Any],
                           mutation_probabity: float,
                           individual_paramgrid: dict, how: str = 'multiple') -> Tuple[Any, Dict[str, Any]]:
        """
        Mutates params of an individual given a mutation probability and param_grid
        :param individual: Individual to mutate
        :param individual_params: individual current parameters
        :param mutation_probabity: Param value mutation probability
        :param individual_paramgrid: Param grid for knowing boundaries and types.
        :param how: single (single param mutation) or multiple (multiple param mutation)
        :return: New individual.
        """
        estimator = individual.__class__
        params = individual_params
        if how == 'multiple':
            for param in params:
                params.update(
                    {param: self.mutate_param(params[param], mutation_probabity, individual_paramgrid[param])}
                )
        elif how == 'single':
            param = random.choice(list(params))
            params.update(
                {param: self.mutate_param(params[param], mutation_probabity, individual_paramgrid[param])}
            )
        else:
            raise ValueError(f'arg how must be either "multiple" or "single". Got {how}')

        return estimator(**params), params

    @staticmethod
    def crossover(*parents: Any, parents_params: Tuple[Dict[str, Any]],
                  num_children: int, how: str) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """
        Performs a crossover between any number of parents and return any number of children
        :param parents: Any Estimators of same kind
        :param parents_params: Parent parameters
        :param num_children: Number of children to generate
        :param how: combination (combination of parents params) or merge (mean aggregation of parents params)
        :return: Children generated.
        """
        assert all(isinstance(p, parents[0].__class__) for p in parents), \
            f'All parents must be from same population. Got {[p.__class__ for p in parents]}'
        assert len(parents) == len(parents_params), (f'parents and parent_params must have same lenght. '
                                                     f'Got {len(parents)} != {len(parents_params)}')
        estimator = parents[0].__class__
        params = {k: [v1] for k, v1 in parents_params[0].items()}
        for parent in parents_params[1:]:
            params.update({k: params[k] + [parent.get(k)] for k in params})
        childs = []
        childs_params = []
        for child in range(num_children):
            if how == 'combination':
                child_params = {
                    param: random.choice(params[param])
                    for param in params
                }
            elif how == 'merge':
                child_params = {}
                for param in params:
                    if any(isinstance(p, (str, bool)) for p in params[param]):
                        child_params.update({
                            param: random.choice(params[param])
                        })
                    elif any(isinstance(p, float) for p in params[param]):
                        child_params.update({
                            param: sum(params[param]) / (len(params[param]))
                        })
                    elif any(isinstance(p, int) for p in params):
                        child_params.update({
                            param: int(sum(params) / (len(params)))
                        })
            else:
                raise ValueError(f'arg how must be either "combination" or "merge". Got {how}')
            try:
                childs.append(estimator(**child_params))
                childs_params.append(child_params)
            except ValueError:
                pass
        return childs, childs_params


class EvolutionaryOptimizer(BaseEvolutionaryOptimizer):
    """
    Evolutionary Algorithms applied to hyperparameter optimization
    """

    def __init__(self,
                 individuals_paramgrid: Dict[type, Dict[str, Any]],
                 population_size: int,
                 selection_size: int,
                 mutation_probability: float,
                 num_parents: int = 2,
                 num_children: int = 1,
                 tournament_size: int = 8,
                 elite_size: Optional[int] = None,
                 commaplus: str = 'comma',
                 crossover_type: str = 'combination',
                 selection_type: str = 'tournament',
                 mutation_type: str = 'multiple',
                 random_state: Optional[int] = None,
                 verbose: int = 0):
        """
        Class constructor
        :param individuals_paramgrid: Parameter grid for each individual family
        :param population_size: Number of individuals in initial population
        :param selection_size: Number of selected individuals after each evaluation (mu)
        :param mutation_probability: Probability of mutating a gene.
        :param num_parents: Number of parents per child
        :param num_children: Number of children per offspring
        :param tournament_size: Numbers of individuals selected in each tournament
        :param elite_size: Number of unaltered best individuals per family in each generation.
        :param commaplus: Select mu (comma) or select mu and lambda (plus)
        :param crossover_type: How parent genes are combined to generate offspring
        :param selection_type: How selection of survivors is made
        :param mutation_type: If single or multiple genes ar mutated
        :param random_state: Deterministic random state.
        :param verbose: Verbosity.
        """
        assert crossover_type in ['combination', 'merge'], \
            f'crossover_type must be one of: "combination", "merge". Got {crossover_type}'
        assert selection_type in ['score', 'tournament'], \
            f'selection_type must be one of: "score", "tournament". Got {selection_type}'
        assert mutation_type in ['single', 'multiple'], \
            f'mutation_type must be one of: "single", "multiple". Got {mutation_type}'
        assert commaplus in ['comma', 'plus'], \
            f'commaplus argument must be one of "comma", "plus". Got {commaplus}'
        if random_state is not None:
            self._set_random_state(random_state)
        self.lambd = population_size
        self.mu = selection_size
        self.commaplus = commaplus
        self.pm = mutation_probability
        self.num_parents = num_parents
        self.num_children = num_children
        self.initial_params = {'lambd': population_size,
                               'mu': selection_size,
                               'pm': mutation_probability,
                               'num_parents': num_parents,
                               'num_children': num_children}
        self.crossover_type = crossover_type
        self.selection_type = selection_type
        self.mutation_type = mutation_type
        self.tournament_size = tournament_size
        if elite_size is not None:
            assert elite_size > 0, f'elite_size must be greater than 0. Got {elite_size}.'
            self.elite_size = elite_size
        else:
            self.elite_size = 0
        self.individuals_paramgrid = individuals_paramgrid
        self.verbose = verbose
        self.scores = {}
        self.fittest_individual = None

    def _reset_parameters(self):
        """
        Resets parameters
        """
        self.lambd = self.initial_params['lambd']
        self.mu = self.initial_params['mu']
        self.pm = self.initial_params['pm']
        self.num_parents = self.initial_params['num_parents']
        self.num_children = self.initial_params['num_children']

    def _check_increase_diversity_dinamically_viability(self,
                                                        patience: Optional[int],
                                                        diversity_increase_factor: Optional[float]
                                                        ) -> Tuple[int, float]:
        """
        Checks if current parameters are suitable for increasing diversity dinamically
        :param patience: patience until early stopping
        :param diversity_increase_factor: factor of diversity increase in each generation
        :return: patience, diversity increase
        """

        assert any([diversity_increase_factor is not None, patience is not None]), \
            'If increasing diversity dinamically you must pass a ' \
            '"diversity_increase_factor" and/or a "patience"'
        if all([diversity_increase_factor is not None, patience is not None]):
            assert 0. < self.pm + patience * diversity_increase_factor < 1., \
                'Current combination of parameters ["mutation_probability", "patience",' \
                ' "diversity_increase_factor" leads into mutation probabilities greater than 1.' \
                'mutation_probability + patience * diversity_increase_factor = ' \
                f'{self.pm} + {patience} * {diversity_increase_factor} = ' \
                f'{self.pm + patience * diversity_increase_factor} \u2284 (0., 1.)'
        if diversity_increase_factor is None:
            diversity_increase_factor = (0.9999 - self.pm) / patience
        if patience is None:
            patience = int((0.9999 - self.pm) / diversity_increase_factor)
        return patience, diversity_increase_factor

    def _increase_diversity(self, diversity_increase: float):
        """
        Increases diversity of the algorithm
        :param diversity_increase: diversity increasing factor
        """
        self.lambd = int(self.lambd + diversity_increase * self.lambd)
        self.mu = int(self.mu + diversity_increase * self.mu)
        self.pm = min(1., self.pm + diversity_increase)
        # More parents, more children
        self.num_parents = (self.mu * self.initial_params['num_parents']) / self.initial_params['mu']
        self.num_children = (self.mu * self.initial_params['num_children']) / self.initial_params['mu']
        self.num_parents = max(1, int(Decimal(self.num_parents).to_integral_exact(ROUND_HALF_UP)))
        self.num_children = max(1, int(Decimal(self.num_children).to_integral_exact(ROUND_HALF_UP)))

    @abc.abstractmethod
    def score_individual(self, individual: Any,
                         *args: Optional[Any], **kwargs: Optional[Any]) -> Tuple[Any, float]:
        """
        Scores a single individual
        :param individual: Individual to score
        :param args: Arguments for scoring
        :param kwargs: Keyword Arguments for scoring
        :return: individual and score (higher is better).
        Something like:
            individual = individual.fit(x_tr, y_tr)
            y_pred = individual.predict(y_val)
            score = mean_absolute_error(y_pred, y_val)
        """
        return individual, float('-inf')

    def score_population(self,
                         population: Dict,
                         individual_params: Dict,
                         families: Dict,
                         *args: Optional[Any],
                         **kwargs: Optional[Any]) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, str]]:
        """
        Scores a single individual
        :param population: Population to score
        :param individual_params: Individual parameters
        :param families: Family mapping
        :param args: Arguments for scoring
        :param kwargs: Keyword Arguments for scoring
        :return: Population and families
        """
        times = []
        with logging_redirect_tqdm():
            pbar = tqdm(enumerate(population),
                        total=len(population),
                        desc='Scoring individual',
                        disable=self.verbose <= 0)
            for i, m in pbar:
                if self.verbose > 0:
                    mscore = np.max(list(self.scores.values())) if len(self.scores) > 0 else 0
                    pbar.set_description(f'Max score: {mscore: .3f}')
                if self.scores.get(m) is None:
                    it = perf_counter()
                    individual, score = self.score_individual(individual=population[m], *args, **kwargs)
                    population.update({m: individual})
                    self.scores.update({m: score})
                    et = perf_counter()
                    times.append(et - it)
        if self.verbose > 1:
            lg.info('Scoring geneneration times:\n'
                    f'\t-Total time: {sum(times): .3f}\n'
                    f'\t-Individual time: {np.mean(times): .3f} \u00B1 {np.std(times): .3f}')

        sorted_scores = {k: v for k, v in dict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True)).items()}
        sorted_population = {i: population[i] for i in sorted_scores}
        population = {i: sorted_population[i] for i in list(sorted_population)[:self.mu]}
        individual_params = {i: individual_params[i] for i in list(sorted_population)[:self.mu]}
        families = {i: families[i] for i in list(sorted_population)[:self.mu]}
        self.scores = {i: sorted_scores[i] for i in list(sorted_scores)[:self.mu]}
        if len(population) < (self.mu if self.commaplus == 'comma' else self.mu + self.lambd):
            refill_population, refill_individual_params, refill_families = self.initialize_population(
                population_size=self.lambd if self.commaplus == 'comma' else self.mu + self.lambd,
                individuals_paramdict=self.individuals_paramgrid,
                population=population,
                individual_params=individual_params,
                families=families
            )
            population.update(refill_population)
            individual_params.update(refill_individual_params)
            families.update(refill_families)
        sc_values = list(self.scores.values())
        if self.verbose > 1:
            lg.info(
                'Survivors stats:\n'
                f'\t-Scores: {np.mean(sc_values): .3f} \u00B1 {np.std(sc_values): .3f}\n'
                f'\t-Max score: {np.max(sc_values): .3f}\n'
                f'\t-Min score: {np.min(sc_values): .3f}\n'
                f'\t-Num individuals: {len(self.scores)}')
        if self.verbose > 0:
            lg.info(f'Best scored model: \n{population[list(population)[0]]} -> '
                    f'Score: {np.max(sc_values): .3f}')
        self.fittest_individual = population[list(population)[0]]
        return population, individual_params, families

    def _update_population(self,
                           population: Dict,
                           individual_params: Dict,
                           families: Dict) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, str]]:
        """
        Update population by generating new individuals by crossover until population size is reached
         then mutate individuals
        :param population: Current population
        :param individual_params: Current individual parameters
        :param families: Current families
        :return: Population and families
        """
        elite = []
        if self.elite_size > 0:
            for f in set(families.values()):
                elite += [idx for idx, ind in population.items() if ind.__class__.__name__ == f][:self.elite_size]
        survivor_population = list(population.values())
        survivor_population_idx = list(population)
        survivor_params = list(individual_params.values())
        survivor_families = list(families.values())
        while len(population) < self.lambd:
            family = random.choice(survivor_families)
            members = [(sp, parm) for sp, parm, f in zip(survivor_population,
                                                         survivor_params,
                                                         survivor_families) if f == family]
            if self.selection_type == 'score':
                parents, parents_params = zip(*random.sample(members,
                                                             k=min(self.num_parents, len(members))))
            elif self.selection_type == 'tournament':
                members_score = [self.scores[i] for i, f in
                                 zip(survivor_population_idx, survivor_families) if f == family]
                tournament = random.sample(range(len(members)),
                                           k=min(self.tournament_size, len(members)))
                weights = [members_score[idx] - min(members_score) for idx in tournament]
                idxs = random.choices(tournament,
                                      weights=weights if sum(weights) > 0. else None,
                                      k=min(self.num_parents, len(members)))
                parents, parents_params = zip(*[members[idx] for idx in idxs])
            else:
                error = ValueError(f'selection_type must be "score" or "tournament". Got {self.selection_type}')
                lg.error(error)
                raise error
            if len(parents) > 0:
                children, children_params = self.crossover(
                    *parents, parents_params=parents_params,
                    num_children=self.num_children, how=self.crossover_type
                )
                if len(children) > 0:
                    for child, child_params in zip(children, children_params):
                        population.update({max(population) + 1: child})
                        individual_params.update({max(individual_params) + 1: child_params})
                        families.update({max(families) + 1: family})
        for i in set(population).difference(elite):
            new_ind, new_params = self.mutate_inidividual(population[i],
                                                          individual_params[i],
                                                          self.pm,
                                                          self.individuals_paramgrid[population[i].__class__],
                                                          how=self.mutation_type)
            population.update({i: new_ind})
            individual_params.update({i: new_params})

        self.scores = {k: self.scores[k] for k in elite if k in self.scores}
        return population, individual_params, families

    def evolution(self,
                  max_generations: int, patience: Optional[int] = None,
                  increase_diversity_dynamically: bool = True,
                  diversity_increase_factor: Optional[float] = 0.1,
                  max_score_val: Optional[float] = None,
                  *score_args: Optional[Any],
                  **score_kwargs: Optional[Any]) -> Any:
        """
        Evolves population init to end.
        :param max_generations: Maximum generations
        :param patience: Maximum generations without improving.
        :param increase_diversity_dynamically: Increase diversity if algorithm stucks.
        :param diversity_increase_factor: diversity increasing factor.
        :param max_score_val: If max_score_val is reached an early stopping is called.
        :param score_args: Arguments for scoring
        :param score_kwargs: Keyword Arguments for scoring
        :return: fittest individual.
        """
        if increase_diversity_dynamically:
            patience, diversity_increase_factor = self._check_increase_diversity_dinamically_viability(
                patience, diversity_increase_factor
            )
        population, individual_params, families = self.initialize_population(
            population_size=self.lambd,
            individuals_paramdict=self.individuals_paramgrid
        )
        generation = 1
        g_no_improve = 0
        max_score = -float('inf')
        while True:
            if self.verbose > 0:
                lg.info(f'Generation: {generation} / {max_generations}. ')
                if patience is not None:
                    lg.info(f'Generations without improvement: {g_no_improve} / {patience}')
            if self.verbose > 1:
                lg.info(f'\t-Population size: {self.lambd}')
                lg.info(f'\t-Num survivors: {self.mu}')
                lg.info(f'\t-Mutation probability: {self.pm: .2f}')
            population, individual_params, families = self.score_population(population,
                                                                            individual_params,
                                                                            families,
                                                                            *score_args, **score_kwargs)
            if max(self.scores.values()) > max_score:
                max_score = max(self.scores.values())
                g_no_improve = 0
                self._reset_parameters()
            else:
                g_no_improve += 1
                if increase_diversity_dynamically:
                    self._increase_diversity(diversity_increase_factor)
            if generation == max_generations:
                break
            if patience is not None and (g_no_improve > patience):
                break
            if max_score_val is not None and max_score >= max_score_val:
                break
            population, individual_params, families = self._update_population(population, individual_params, families)
            generation += 1
        return population[list(population)[0]]

    def __call__(self,
                 max_generations: int = 50, patience: Optional[int] = None,
                 increase_diversity_dynamically: bool = True,
                 diversity_increase_factor: Optional[float] = None,
                 max_score_val: Optional[float] = None,
                 *score_args: Optional[Any],
                 **score_kwargs: Optional[Any]
                 ):
        try:
            model = self.evolution(max_generations=max_generations, patience=patience,
                                   increase_diversity_dynamically=increase_diversity_dynamically,
                                   diversity_increase_factor=diversity_increase_factor,
                                   max_score_val=max_score_val,
                                   *score_args, **score_kwargs)
        except KeyboardInterrupt:
            lg.error('Evolution strategy aborted. Returning best.')
            model = self.fittest_individual
        return model
