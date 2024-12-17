from typing import Dict, Any, Tuple, Optional, List
from dataclasses import asdict, dataclass, field
import logging as lg
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from evolutionary_hp_optim.evolutionary_estimator import EvolutionaryEstimator

logger = lg.getLogger(__name__)


@dataclass
class ExtraTreesClassifierParams:
    n_estimators: List[int] = field(default_factory=lambda: [10, 50, 100, 250])
    criterion: List[str] = field(default_factory=lambda: ['gini'])
    max_depth: List[int] = field(default_factory=lambda: [None])
    min_samples_split: List[int] = field(default_factory=lambda: [2, 10, 50, 100])
    min_samples_leaf: List[int] = field(default_factory=lambda: [1, 5, 25, 50])
    min_weight_fraction_leaf: List[float] = field(default_factory=lambda: [0., 0.1, 0.25, 0.49])
    max_features: List[Optional[str]] = field(default_factory=lambda: ['sqrt', 'log2', None])
    max_leaf_nodes: List[int] = field(default_factory=lambda: [None])
    min_impurity_decrease: List[float] = field(default_factory=lambda: [0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    bootstrap: List[bool] = field(default_factory=lambda: [True, False])
    oob_score: List[bool] = field(default_factory=lambda: [False])
    n_jobs: List[Optional[int]] = field(default_factory=lambda: [-1])
    random_state: List[int] = field(default_factory=lambda: [0])
    verbose: List[int] = field(default_factory=lambda: [0])
    warm_start: List[bool] = field(default_factory=lambda: [False])
    class_weight: List[str] = field(default_factory=lambda: ['balanced', 'balanced_subsample'])
    ccp_alpha: List[float] = field(default_factory=lambda: [0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    max_samples: List[int] = field(default_factory=lambda: [None])


@dataclass
class GradientBoostingClassifierParams:
    loss: List[str] = field(default_factory=lambda: ['deviance'])
    learning_rate: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2, 0.1])
    n_estimators: List[int] = field(default_factory=lambda: [10, 50, 100, 250])
    subsample: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    criterion: List[str] = field(default_factory=lambda: ['friedman_mse', 'squared_error'])
    min_samples_split: List[int] = field(default_factory=lambda: [2, 10, 50, 100])
    min_samples_leaf: List[int] = field(default_factory=lambda: [1, 5, 25, 50])
    min_weight_fraction_leaf: List[float] = field(default_factory=lambda: [0., 0.1, 0.25, 0.49])
    max_depth: List[int] = field(default_factory=lambda: [None])
    min_impurity_decrease: List[float] = field(default_factory=lambda: [0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    init: List[Optional[str]] = field(default_factory=lambda: [None])
    random_state: List[int] = field(default_factory=lambda: [0])
    max_features: List[Optional[str]] = field(default_factory=lambda: ['sqrt', 'log2', None])
    verbose: List[int] = field(default_factory=lambda: [0])
    max_leaf_nodes: List[int] = field(default_factory=lambda: [None])
    warm_start: List[bool] = field(default_factory=lambda: [False])
    validation_fraction: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5])
    n_iter_no_change: List[Optional[int]] = field(default_factory=lambda: [None])
    tol: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2, 0.1])
    ccp_alpha: List[float] = field(default_factory=lambda: [0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])


@dataclass
class RandomForestClassifierParams:
    n_estimators: List[int] = field(default_factory=lambda: [10, 50, 100, 250])
    criterion: List[str] = field(default_factory=lambda: ['gini'])
    max_depth: List[int] = field(default_factory=lambda: [None])
    min_samples_split: List[int] = field(default_factory=lambda: [2, 10, 50, 100])
    min_samples_leaf: List[int] = field(default_factory=lambda: [1, 5, 25, 50])
    min_weight_fraction_leaf: List[float] = field(default_factory=lambda: [0., 0.1, 0.25, 0.49])
    max_features: List[Optional[str]] = field(default_factory=lambda: ['sqrt', 'log2', None])
    max_leaf_nodes: List[int] = field(default_factory=lambda: [None])
    min_impurity_decrease: List[float] = field(default_factory=lambda: [0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    bootstrap: List[bool] = field(default_factory=lambda: [True, False])
    oob_score: List[bool] = field(default_factory=lambda: [False])
    n_jobs: List[Optional[int]] = field(default_factory=lambda: [-1])
    random_state: List[int] = field(default_factory=lambda: [0])
    verbose: List[int] = field(default_factory=lambda: [0])
    warm_start: List[bool] = field(default_factory=lambda: [False])
    class_weight: List[str] = field(default_factory=lambda: ['balanced', 'balanced_subsample'])
    ccp_alpha: List[float] = field(default_factory=lambda: [0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    max_samples: List[int] = field(default_factory=lambda: [None])


@dataclass
class DecisionTreeClassifierParams:
    criterion: List[str] = field(default_factory=lambda: ['gini'])
    splitter: List[str] = field(default_factory=lambda: ['best', 'random'])
    max_depth: List[int] = field(default_factory=lambda: [None])
    min_samples_split: List[int] = field(default_factory=lambda: [2, 10, 50, 100])
    min_samples_leaf: List[int] = field(default_factory=lambda: [1, 5, 25, 50])
    min_weight_fraction_leaf: List[float] = field(default_factory=lambda: [0., 0.1, 0.25, 0.49])
    max_features: List[Optional[str]] = field(default_factory=lambda: ['sqrt', 'log2', None])
    random_state: List[int] = field(default_factory=lambda: [0])
    max_leaf_nodes: List[int] = field(default_factory=lambda: [None])
    min_impurity_decrease: List[float] = field(default_factory=lambda: [0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    class_weight: List[Optional[str]] = field(default_factory=lambda: ['balanced', None])
    ccp_alpha: List[float] = field(default_factory=lambda: [0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])


@dataclass
class ExtraTreeClassifierParams:
    criterion: List[str] = field(default_factory=lambda: ['gini'])
    splitter: List[str] = field(default_factory=lambda: ['best', 'random'])
    max_depth: List[int] = field(default_factory=lambda: [None])
    min_samples_split: List[int] = field(default_factory=lambda: [2, 10, 50, 100])
    min_samples_leaf: List[int] = field(default_factory=lambda: [1, 5, 25, 50])
    min_weight_fraction_leaf: List[float] = field(default_factory=lambda: [0., 0.1, 0.25, 0.49])
    max_features: List[Optional[str]] = field(default_factory=lambda: ['sqrt', 'log2', None])
    random_state: List[int] = field(default_factory=lambda: [0])
    max_leaf_nodes: List[int] = field(default_factory=lambda: [None])
    min_impurity_decrease: List[float] = field(default_factory=lambda: [0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    class_weight: List[Optional[str]] = field(default_factory=lambda: ['balanced', None])
    ccp_alpha: List[float] = field(default_factory=lambda: [0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])


class MyEstimator(EvolutionaryEstimator):
    def __init__(self, population_size: int, individuals_paramgrid: Dict[type, Dict[str, Any]],
                 selection_size: int,
                 mutation_probability: float,
                 crossover_type: str = 'combination',
                 selection_type: str = 'score',
                 mutation_type: str = 'multiple',
                 comma_plus: str = 'comma',
                 elite_size: Optional[int] = 1,
                 num_parents: int = 4,
                 num_children: int = 2,
                 tournament_size: int = 8,
                 verbose: int = 1):
        super(). __init__(population_size, individuals_paramgrid, selection_size, mutation_probability,
                          crossover_type, selection_type, mutation_type, comma_plus, elite_size,
                          num_parents, num_children, tournament_size, verbose)

    def score_individual(self, individual: Any,
                         *args: Optional[Any], **kwargs: Optional[Any]) -> Tuple[Any, float]:
        try:
            individual = individual.fit(kwargs.get('x_tr'), kwargs.get('y_tr'))
        except KeyError as e:
            error = f'KeyError: {individual} error.' + repr(e)
            lg.error(error)
            raise e
        except ValueError as e:
            error = f'ValueError: {individual} error.' + repr(e)
            lg.error(error)
            raise e
        y_pred = individual.predict(kwargs.get('x_val'))
        score = roc_auc_score(kwargs.get('y_val'), y_pred)
        return individual, score


if __name__ == '__main__':
    lg.basicConfig(level=lg.INFO)
    console_handler = lg.StreamHandler()
    console_handler.setLevel("INFO")
    logger.addHandler(console_handler)

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=100, random_state=0)
    paramgrid = {
        ExtraTreeClassifier: asdict(ExtraTreeClassifierParams()),
        DecisionTreeClassifier: asdict(DecisionTreeClassifierParams()),
        ExtraTreesClassifier: asdict(ExtraTreesClassifierParams()),
        RandomForestClassifier: asdict(RandomForestClassifierParams()),
        GradientBoostingClassifier: asdict(GradientBoostingClassifierParams())
    }

    me = MyEstimator(population_size=200,
                     individuals_paramgrid=paramgrid,
                     mutation_probability=0.05,
                     selection_size=100,
                     num_parents=2,
                     num_children=1,
                     elite_size=5,
                     crossover_type='combination',
                     comma_plus='comma',
                     selection_type='tournament',
                     verbose=1)
    model = me(x_tr=X_tr, x_val=X_val, y_tr=y_tr, y_val=y_val,
               patience=25, max_score_val=1.)
