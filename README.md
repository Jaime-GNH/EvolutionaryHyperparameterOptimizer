# Evolutionary Hyperparameter Optimizer

Evolution Strategies and genetic algorithms for hyperparameter tunning.

## 1. Use

Some usages are shown in [examples](examples).
1. Classification problem with sklearn common models: [classifier_usage](examples/classifier_usage.py)
2. Regression problem with sklearn common models: [regression_usage](examples/regression_usage.py)
3. [ ] torch.nn.Module tunning.
4. [ ] tf.Keras.Model tunning.

The common usage is done subclassing the [`EvolutionaryOptimizer` class](evolutionary_hp_optim/evolutionary_hp_optimizer.py)
and adding a `score_individual` function.

```python

def score_individual(self, individual: Any,
                     *args, **kwargs) -> Tuple[Any, float]:
    """
    Scores a single individual
    :param individual: Individual to score
    :param args: Arguments for scoring
    :param kwargs: Keyword Arguments for scoring
    :return: individual and score (higher is better).
    Something like:
    """
    individual = individual.fit(args[0], args[1])
    y_pred = individual.predict(args[2])
    score = some_scoring_function(y_pred, args[3])
    return individual, float('-inf')
```

Then you must define a parameter grid as a seed for each family in the population by constructing a dictionary 
`type: Dict[str: List]` that caracterizes a family boundaries and initial potential values.

```python
paramdict = {
    MyTorchNnModule: {'num_layers': [1, 3, 5, 7],
                      'h_dim': [16, 32, 64, 128],
                      'activation_type': ['relu', 'prelu', 'leakyrelu']},
    ...
}
```

Finally, define your evolutionary strategy considering the time and computation requirements and perform the evolution.

```python
fittest_model = MyEvolStrat(...)()
```

## 2. Evolution Strategies

There are multiple evolution strategies inside the global class
[`EvolutionaryOptimizer`](evolutionary_hp_optim/evolutionary_hp_optimizer.py).

Let's see some of them.

### 2.1. Comma / Plus

An algorithm is called comma ($$\left(\mu,\lambda\right)$$) if the population is replaced with the
offspring in each generation.

An algorithm is called plus ($$\left(\mu+\lambda\right)$$) if the offspring is added to the surviving population 
in each generation.

### 2.2. Crossover type

The `crosover_type` is how a crossover is made between parents. Two options are given:

1. Combination: Each parameter defining a new model (child) is given as a random selection from parents.
2. Merge: Each parameter defining a new model (child) is given as an aggregation of this parents' value.

Combination explores more, merge narrows the search.

### 2.3. Selection type

The `selection_type` defines how the parents for the new generation (*the survivors*) are selected.

1. Score: Ordering by simple scoring the top-k (`selection_size`) are selected.
2. Tournament: The top-1 score of a tournament of N individuals is selected until `selection_size` individuals are chosen.

Tournament allows more exploration than score. Tournament should be used along `elite_size` for stability.

### 2.4. Mutation type

An individual can mutate a `single` parameter (*gene*) or `multiple` parameters at once.

A `single` `mutation_type` increases stability but lowers exploration so it can lead to local minima's.
In contrast, `multiple` `mutation_type` increases exploration but can skip global minima's easily.

As a rule of thumb consider:

- If you have a good and narrow initial search space: choose `single`.
- If you want to narrow your initial solution: choose `multiple`.
- If you are choosing `multiple` keep low (<0.1) the `mutation_probability` and increment `elite_size` for greater stability.

### 2.5. Diversity increase

In the `__call__` function you can impose the *dinamical diversity increase* mode that will enlarge the exploration capabilities
of your algorithm if the score is getting better.

This argument addresses the flexibility issue of a fixed evolution strategy allowing the evolution to reach the corners 
when is needed.


### 2.6. All combinations

This leads into 32 different evolution strategies that balance more or less the exploration/explotation rate.

Moreover, there are other parameters that define the proceding as:

- num_childs: Number of children per set of parents in each generation.
- num_parents: Number of parents involved in generating a child.
- elite_size: Number of individuals per family that are mantained without any mutation.
- selection_size: Number of indiviuduals surviving the scoring.

## 3. Version

### 3.1. Version 0.0.0.1 

Initial version with base functionalities and documents.

Some use cases added.

- [ ] Extend to torch and keras.
- [ ] Add flexibility and modes to EvolutionaryOptimizer.
- [ ] Add more conditional Early Stoppings and exploration boosters.
- [ ] Create usages in Jupyer notebooks.
- [ ] Listen to people requirements and propositions.


