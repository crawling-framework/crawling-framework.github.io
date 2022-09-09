======================
Installation and usage
======================

Requirements
------------

* python version 3
* GCC compiler
* cmake

For MacOS additional:

* brew

Install
-------

1. Compile C++ code
::
   make


2. Install python dependencies
::
   pip install -r requirements.txt


.. See the documentation and tutorials at https://crawling-framework.github.io/

DGL with CPU/GPU https://www.dgl.ai/pages/start.html

Use
---

Run 1 crawler from command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Run from src/ folder:
::
   python experiments/cmd.py -g <GRAPH> -c <CRAWLER> -n <RUNS>

To see available options type:: `python experiments/cmd.py -h`

Reproduce experiments from the paper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To obtain all the results from Table 4 one can run all configurations:
::
   python experiments/paper_experiments.py

but this can take very long time (up to several weeks).
Edit the file `paper_experiments.py` to run a proper configuration.

Once a crawler has finished its job, its result is saved to a corresponding file in `results/` folder.
Script `python experiments/paper_plots.py` will collect statistics of all the computed results.
