func_mining
==============================

Code for the paper "Mining within-trial oscillatory brain dynamics to address the variability of optimized spatial filters". The paper is soon available here.

How to run the example notebook
------------

The code was tested on Python 3.5.2, other version may work as well, but we recommend using a version >=3.5.2. Note that the following instructions are written for users using a default Python3 installation. Commands may differ for AnaConda users.

#### 0. Set up a virtual environment (recommended) ####

It's always a good idea to set up a virtual environment for new projects. In order to create a virtual environment, change to the directory in which you cloned this repository and execute:

    python3 -m venv name_of_your_venv_folder

Now activate your environment:

    source name_of_your_venv_folder/bin/activate

Use `deactivate` to leave the virtual environment after you are done with this notebook / repository.

#### 1. Download the provided data set ####

Run ``make data`` in the root folder of the repository.

Alternatively, download the provided exemplary features set of an
exemplary subject available on
[zenodo](https://zenodo.org/record/1237814#.WucuW9a-lhE).  Unzip the
data and place the "processed" folder under ./data of this repository.

#### 2. Install requirements ####

    pip install -r requirements.txt

#### 3. Run the jupyter notebook ####

Make sure you are currently in the repository root and then:

    jupyter notebook notebooks/Mining_functional_brain_signals.ipynb

The notebook should open automatically in your default web browser. If it does not do so, visit `localhost:8888` in your favourite browser.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── processed      <- Exemplary dataset of different spatial filters computed on data of one subject 
    │    
    ├── notebooks          <- Example jupyter notebook for using our proposed method. 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip install -r requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Import component data
    │   │    
    │   ├── features       <- Feature preprocessing for clustering    │   │  
    │   │
    │   ├── models         <- DBSCAN clustering
    │   │     
    │   └── visualization  <- Varioous tools to inspect / characterize the clustering results
    │                        
    │
    └── .gitignore         <- list of files to ignore for git 


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
