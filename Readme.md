# Network crawling framework

The CrawlingFramework is aimed for offline testing of network crawling algorithms on social graphs. 
Undirected graphs without self-loops are supported.

Currently, framework allows reproducing experiments from the paper "Amplifying Online Learning with 
Graph Neural Networks for Selective Harvesting over Social Networks" (not published).

**Features**:
* Run crawlers on given graphs, calculating quality measures, saving the history, and drawing 
plots.

## Installation

1. Compile C++ code
```
make
```

2. Install python dependencies
```
pip install -r requirements.txt
```

See the documentation and tutorials at https://crawling-framework.github.io/

DGL with CPU/GPU https://www.dgl.ai/pages/start.html

## Usage

### Run 1 crawler from command line

``` python experiments/cmd.py -g <GRAPH> -c <CRAWLER> -n <RUNS>```

To see available options type ` python experiments/cmd.py -h`

### Reproduce experiments from the paper

To obtain all the results from Table 4 one can run all configurations:

```python experiments/paper_experiments.py```

but this can take very long time (up to several weeks).
Edit the file `paper_experiments.py` to run a proper configuration.

Once a crawler has finished its job, its result is saved to a corresponding file in `results/` folder.
Script `python experiments/paper_plots.py` will collect statistics of all the computed results.
