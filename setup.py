from setuptools import setup

setup(
    name='EvolutionaryHyperparameterOptimizer',
    version='0.0.0.1',
    install_requires=[
        # Pip
        'pip>=24.0',
        'setuptools>=75.6.0',
        'tqdm>=4.67.1',
        'numpy>=2.2.0'
    ],
    packages=['evolutionary_hp_optim'],
    license='',
    author='Jaime Gonzalez-Novo',
    author_email='jaimegonzaleznovohueso@gmail.com',
    description=u'Genetic Algorithms for sklearn models param optimization.'
)
