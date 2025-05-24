# ROM-RL-pub
Code for the paper "Interpretable and Efficient Data-Driven Discovery and Control of Distributed Systems"
by Florian Wolf, Nicolò Botteghi, Urban Fasel, and Andrea Manzoni.
The preprint is available on [arxiv](https://arxiv.org/abs/2411.04098).

To simply reproduce all experiments in the paper, execute
```shell
cd src
sh run_experiments.sh
```
in your console.

## Structure of the code

- ``/SindyRL-AE``: This sub-folder is based on the [sindy-rl](https://github.com/nzolman/sindy-rl) code
    and the corresponding paper "SINDy-RL: Interpretable and Efficient
    Model-Based Reinforcement Learning" by Zolman et al., see [arxiv-link](https://arxiv.org/abs/2403.09110).
    Check ``/SindyRL-AE/documents/LICENSE.pdf``.

    We extended the code by the following new functionalities to enable the support of PDE experiments
    and our auto encoder framework:

    - ``dynamics.py`` containing the ``AutoEncoderDynamicsModel(BaseDynamicsModel)``,
    - ``burgers.py`` wrapping the ``controlgym`` environment for the first experiment of the paper,
    - ``navier_stokes.py`` wrapping ``PDEControlGym`` environment for the second experiment of the paper,

- ``/src``: This sub-folder contains the actual configuration files, training and evaluation scripts.
    - ``run_experiments.sh`` will run all experiments present in the paper.
    - ``/config_templates`` contains the configuration files for the experiments in the paper.
    Each of the files corresponds to exactly one of the experiments and models shown in the paper.
    - ``/autoencoder`` provides the implementation of the auto encoder model, loss function and a custom
    logger.
    - ``/analysis``: Helper methods to read logs, count parameters and create the heatmaps presented
    in the paper. 
    - ``sindyrl_<>.py`` are the main scripts to train, load and evaluate the models presented in the
    paper.


## Installation guideline
The easiest solution is to directly use the provided conda environment, via:
```
conda env create --name sindyrl --file=env_sindyrl.yml
conda activate sindyrl
```
or via the docker installation instructions provided in the ``sindy-rl`` repo.

Alternatively, one can manually install the dependencies
- Create a virtual environment with ``python 3.10.14``.
- Install requirements of ``sindyrl`` via
    ```shell
    cd sindy_rl
    pip install -r requirements.txt
    ```
- Installation of ``controlgym`` from https://github.com/Flo-Wo/controlgym via
    ```shell
    git clone git@github.com:Flo-Wo/controlgym.git
    cd controlgym
    pip install -r .
    ```
- Install ``PDEcontrolGym`` from https://github.com/lukebhan/PDEControlGym.
- Install additional requirements via ``pip install -r requirements.txt``.
