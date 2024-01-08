## How to install environment and run project for the first time

1. Install env with
```shell
conda env create -f environment.yml
```

2. Activate env with
```shell
conda activate brec-env
```

## How to update already installed environment

```shell
conda activate brec-env
```
```shell
conda env update --file environment.yml --prune
```

## How to add new dependencies?

To add new dependencies, either 
```shell
conda install <packagename>
```
or
```shell
pip install <packagename>
```

After intalling new dependencies, the environment.yml has to be updated. It can be updated by using:

```shell
conda env export > temp_environment.yml
```

This will export the environment with all os specific packages and it also includes all packages installed via pip. Next, run

```shell
conda env export > environment.yml --from-history
```

To create an env that contains packages that you have installed by yourself only. However, it does not include packages installed with pip. For that, copy everything under `dependencies: -pip` in `temp_environment.yml` into `environment.yml`. Afterwards, delete `temp_environment.yml`.

## Project working routine

To work and use this project, one needs to consider the following steps.

Update the `environment.yml` file by adding your additional packages.

## Problems to work with the environment.yml? 
In case you have problems working with the environment.yml file and need to create the environment yourself follow stepts 2 and 3 and install missing packages manuelly. However, assume everything works fine, then install environment as descibred above.

2. Install conda for Python 3.8.13 according to the 
   [installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

3. Create a conda environment with the name `brec-env`.
```shell
conda create --name brec-env python=3.8.13
```
4. Install missing packages. (Very important: PyTorch 1.13.1 + PyTorch_Geometric 2.2). Other required Python libraries included: numpy, networkx, loguru, etc.

For reproducing other results, please refer to the corresponding requirements for additional libraries. see: [project guide](https://github.com/brec-iclr2024/brec-iclr2024) 

## Comments to myself :D
What is the difference between this and the one above?
In case you installed new packages, always update the `environment.yml` file by
```shell
conda env export | grep -v "^prefix: " > environment.yml
```
