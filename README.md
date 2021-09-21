# Hindsight Foresight Relabeling for Meta-Reinforcement Learning

arxiv link: https://arxiv.org/abs/2109.09031

by Michael Wan, Jian Peng, and Tanmay Gangwani

> Meta-reinforcement learning (meta-RL) algorithms allow for agents to learn new behaviors from small amounts of experience, mitigating the sample inefficiency problem in RL. However, while meta-RL agents can adapt quickly to new tasks at test time after experiencing only a few trajectories, the meta-training process is still sample-inefficient. Prior works have found that in the multi-task RL setting, relabeling past transitions and thus sharing experience among tasks can improve sample efficiency and asymptotic performance. We apply this idea to the meta-RL setting and devise a new relabeling method called Hindsight Foresight Relabeling (HFR). We construct a relabeling distribution using the combination of "hindsight", which is used to relabel trajectories using reward functions from the training task distribution, and "foresight", which takes the relabeled trajectories and computes the utility of each trajectory for each task. HFR is easy to implement and readily compatible with existing meta-RL algorithms. We find that HFR improves performance when compared to other relabeling methods on a variety of meta-RL tasks.

This code is built on top of the open-source PEARL (Rakelly et al., 2019) implementation found here: https://github.com/katerakelly/oyster.

## Installation

### Install MuJoCo

1. Obtain a 30-day free trial or license on the [MuJoCo website](https://www.roboti.us/license.html). You will be emailed a file called `mjkey.txt`.
2. Download the MuJoCo version 1.5 and 2.0 binaries.
3. Unzip the downloaded `mjpro150` directory into `~/.mujoco/mjpro150`, unzip the downloaded `mujoco200` directory into ~/.mujoco/mujoco200
   and place `mjkey.txt` at `~/.mujoco/mjkey.txt`.
4. ``
pip install -U 'mujoco-py<1.50.2,>=1.50.1'
`` for MuJoCo version 1.5 or ``pip install -U 'mujoco-py<2.1,>=2.0'`` for MuJoCo version 2.0
MuJoCo version 2.0 was used for the Sawyer environments, while version 1.5 was used for the other environments.

### Install Meta-World
We use Meta-World for the Sawyer environments. Meta-World can be installed with the following:
```
git clone https://github.com/rlworkgroup/metaworld.git
git checkout 2f228c2
cd metaworld
pip install -e .
```
### Install everything else

``
pip install -r requirements.txt
``

This code was tested using Python version 3.6.8.

## Usage
``
python launch_experiment.py <config file>
``
The output directory can be configured with the "base_log_dir" parameter in configs/default.py.

### Example: Running Ant-Goal
``
python launch_experiment.py ./configs/ant-goal.json
``
