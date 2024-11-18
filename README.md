# Deep Reinforcement Learning with OpenFOAM

## Overview

### Repository structure

- **docs**: Sphinx documentation sources (work in progress)
- **drlfoam**: Python library for DRL with OpenFOAM
- **examples**: annotated scripts for performing DRL trainings and tests
- **openfoam**: OpenFOAM simulation setups (*test_cases*) and source files for additional OpenFOAM library components, e.g., boundary conditions (*src*)

For a list of research projects employing drlFoam, refer to the [references](./references.md).

### drlFoam package

- **drlfoam.environment**: wrapper classes for manipulating and parsing OpenFOAM test cases
- **drlfoam.agent**: policy and value networks, agent logic and training
- **drlfoam.execution**: trajectory buffer, local and cluster (Slurm) execution

## Installation

### Python environment

A miss-match between the installed Python-frontend of PyTorch and the C++ frontend (libTorch) can lead to unexpected behavior. Therefore, it is recommended to create a virtual environment using [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/miniconda.html). Moreover, the currently used PyTorch version (1.12.1) requires a Python interpreter **>=3.8**. The following listing shows all steps to set up a suitable virtual environment on Ubuntu 20.04 or newer.
```
# install venv
sudo apt update && sudo apt install python3.8-venv
# if not yet cloned, get drlfoam
git clone https://github.com/OFDataCommittee/drlfoam.git
cd drlfoam
# create a new environment called pydrl
python3 -m venv pydrl
# start the environment and install dependencies
source pydrl/bin/activate
pip install -r requirements.txt
# once the work is done, leave the environment
deactivate
```

### OpenFOAM library components

Source code and test cases are only compatible with **OpenFOAM-v2206**; [installation instructions](https://develop.openfoam.com/Development/openfoam/-/wikis/precompiled). You can use the pre-compiled binaries. Make sure that the OpenFOAM environment variables are available, e.g., by adding `source /usr/lib/openfoam/openfoam2206/etc/bashrc` to `$HOME/.bashrc`. The additional OpenFOAM library components are compiled as follows:
```
# at the top-level of this repository
# when executing for the first time, libTorch is downloaded by Allwmake
source setup-env
./Allwmake
```
In case you want to re-compile starting from a clean state:
```
# assuming you are at the repository's top folder
./Allwclean
```

### Working with Singularity containers

Instead of installing dependencies manually, you can also employ the provided Singularity container. Singularity simplifies execution on HPC infrastructures, because no dependencies except for Singularity itself and OpenMPI are required. To build the container locally, run:
```
sudo singularity build of2206-py1.12.1-cpu.sif docker://andreweiner/of_pytorch:of2206-py1.12.1-cpu
```
By default, the container is expected to be located at the repository's top level. The default location may be changed by adjusting the `DRL_IMAGE` variable in *setup-env*. To build the OpenFOAM library components, provide the `--container` flag:
```
./Allwmake --container
```
Similarly, for cleaning up the build:
```
./Allwclean --container
```

## Running a training

Currently, there are two examples of assembling a DRL training with drlFoam:
1. the *rotatingCylinder2D* test case
2. the *rotatingPinball2D* test case

To perform the training locally, execute the following steps:
```
# from the top-level of this repository
source pydrl/bin/activate
source setup-env
cd examples
# see config_orig.yml for all available options
# defaults to: training saved in test_training; buffer size 4; 2 runners
# this training requires 4 MPI ranks on average and two loops
# of each runner to fill the buffer
python3 run_training.py
```
The settings can be adjusted in the `config_orig.yml`, located in the `examples` directory.
To run the training with the Apptainer container, pass the `--container` flag to *setup-env*:
```
source setup-env --container
python3 run_training.py
```

## Running a training with SLURM

This section describes how to run a training on a HPC with SLURM. The workflow was tested on TU Braunschweig's [Pheonix cluster](https://www.tu-braunschweig.de/en/it/dienste/21/phoenix) and might need small adjustments for other HPC configurations. The cluster should provide the following modules/packages:
- Apptainer
- Python 3.8
- OpenMPI v4.1 (minor difference might be OK)
- SLURM

After logging in, the following steps set up all dependencies. These steps have to be executed only once in a new environment:
```
# load git and clone repository
module load comp/git/latest
git clone https://github.com/OFDataCommittee/drlfoam.git
# copy the Singularity image to the drlfoam folder
cp /path/to/of2206-py1.12.1-cpu.sif drlfoam/
# set up the virtual Python environment
module load python/3.8.2
python3 -m venv pydrl
source pydrl/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# compile the OpenFOAM library components
module load singularity/latest
./Allwmake --container
```
The *examples/run_training.py* scripts support SLURM-based. To run a new training on the cluster, navigate to the *examples* folder and create a new dedicated jobscript, e.g., *training_jobscript*. A suitable jobscript looks as follows:
```
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --job-name=drl_train
#SBATCH --ntasks-per-node=1

module load python/3.8.2

# adjust path if necessary
source ~/drlfoam/pydrl/bin/activate
source ~/drlfoam/setup-env --container

# start a training with a buffer size of 8 and 8 runners;
# save output to log.test_training
python3 run_training.py &> log.test_training
```
Submitting, inspecting, and canceling of trainings works as follows:
```
# start the training
sbatch training_jobscript
# check the SLURM queue
squeue -u $USER
# canceling a job
scancel job_id
```
To detect potential errors in case of failure, inspect *all* log files:
- the log file of the driver script, e.g., *log.test_training*
- the output of individual simulation jobs, e.g., slurm-######.out
- the standard OpenFOAM log files located in *test_training/copy_#*

## Development

Unittests are implemented with [PyTest](https://docs.pytest.org/en/7.1.x/). Some tests require a [Slurm](https://slurm.schedmd.com/documentation.html) installation on the test machine. Instructions for a minimal Slurm setup on Ubuntu are available [here](https://gist.github.com/ckandoth/2acef6310041244a690e4c08d2610423). If Slurm is not available, related tests are ignored. Some test require additional test data, which can be created with the *create_test_data* script.
```
# examples for running all or selected tests
# starting from the repository top-level
# run all available tests with additional output (-s)
source setup-env
pytest -s drlfoam
# run all tests in the agent sub-package
pytest -s drlfoam/agent
# run all tests for the ppo_agent.py module
pytest -s drlfoam/agent/tests/test_ppo_agent.py
```

## Contributions

drlFoam is currently developed and maintained by Andre Weiner (@AndreWeiner). Significant contributions to usability, bug fixes, and tests have been made during the first [OpenFOAM-Machine Learning Hackathon](https://github.com/OFDataCommittee/OFMLHackathon) by (in alphabetical order):

- Ajay Navilarekal Rajgopal (@ajaynr)
- Darshan Thummar (@darshan315)
- Guilherme Lindner (@guilindner)
- Julian Bissantz (@jbissantz)
- Morgan Kerhouant
- Mosayeb Shams (@mosayebshams)
- Tomislav Marić (@tmaric)

The foundation of the drlFoam implementation are the student projects by [Darshan Thummar](https://github.com/darshan315/flow_past_cylinder_by_DRL) and [Fabian Gabriel](https://github.com/FabianGabriel/Active_flow_control_past_cylinder_using_DRL).

## License

drlFoam is [GPLv3](https://en.wikipedia.org/wiki/GNU_General_Public_License)-licensed; refer to the [LICENSE](https://github.com/OFDataCommittee/drlfoam/blob/main/LICENSE) file for more information.