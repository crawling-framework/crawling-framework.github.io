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

Installation
------------

1. Clone the repository
::
   git clone https://github.com/crawling-framework/crawling-framework.github.io.git
   cd crawling-framework.github.io


2. Compile C++ code
::
   make


3. Install python dependencies
::
   pip install -r requirements.txt

.. See the documentation and tutorials at https://crawling-framework.github.io/

(By default DGL library is configured for CPU. To use it with GPU visit https://www.dgl.ai/pages/start.html)

4. Download and unpack archive with graph data
::
   wget https://disk.yandex.ru/d/Z-fcweFaVtBsFA
   unzip data.zip

Usage
-----

The first step is to add `src` directory to the python path:
::
   export PYTHONPATH=src

Run 1 crawler from command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run from project folder:
::
   python src/experiments/cmd.py -g <GRAPH> -c <CRAWLER> -n <RUNS>

At the very first run it takes some time to compile cython code.

To see available options type:: `python experiments/cmd.py -h`

Reproduce experiments from the WSDM23 paper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To obtain all the results from Table 4 one can run all configurations:
::
   python src/experiments/paper_experiments.py

but this can take very long time (up to several weeks).
Edit the file `paper_experiments.py` to run a proper configuration.

Once a crawler has finished its job, its result is saved to a corresponding file in `results/` folder.
Script `python src/experiments/paper_plots.py` will collect statistics of all the computed results.
