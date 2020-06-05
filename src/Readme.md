# Network crawling framework

_description_

**Features**:
* Automatic graphs downloading from [Konect](http://konect.uni-koblenz.de/networks/) and 
[networkrepository](http://networkrepository.com/) online collections.
* Graph statistics (including centralities) can be calculated and are stored together with the
 graphs.
* Several families of crawlers implemented:
  * popular ones: RandomCrawler, RandomWalkCrawler, BreadthFirstSearchCrawler, 
  DepthFirstSearchCrawler, MaximumObservedDegreeCrawler, PreferentialObservedDegreeCrawler;
  * multicrawler mode - when a set of crawlers work together;
  * advanced crawlers - for more complex strategies.
* Run crawlers on given graphs, calculating quality measures, saving the history, and drawing 
plots.

**Planning**:
* graph models: controlled assortativity, ERGG.

## Requirements

* python version 3
* GCC compiler

For MacOS additional:

* brew

## Install

#### For Linux

* Install all needed python libraries (from project folder):
```
pip3 install -r requirements.txt
```
* Install [SNAP](https://snap.stanford.edu/snap/index.html) (in any directory):
```
git clone https://github.com/snap-stanford/snap.git
cd snap
```
In `Makefile.config` add `-fPIC` compiler option: find a string 
`CXXFLAGS += -O3 -DNDEBUG -fopenmp` 
and replace it with
`CXXFLAGS += -O3 -DNDEBUG -fopenmp -fPIC`

Now build it:
```
make all
```

#### For MacOS

* Install all needed python libraries (from project folder):
```
brew install -r requirements.txt
```
* Install [SNAP](https://snap.stanford.edu/snap/index.html) (in any directory):
```
git clone https://github.com/snap-stanford/snap.git
cd snap
make all
```

### Final preparations 

Copy the file `config.exmaple` to `config` - this file will contain your specific flags and paths.
Find the line

`SNAP_DIR = "/path/to/snap"         # directory with snap built`

Put there your path to the installed snap root directory.
(NOTE: don't start the path from '~' or it will fail with the g++ option '-I')

## Usage

One may toggle several switchers:

* In file `src/utils.py` set `USE_CYTHON_CRAWLERS = True` -
to use cython-optimized version.

* In file `src/statistics.py` set `USE_NETWORKIT = True` - 
to use [Networkit](https://networkit.github.io/) library to compute centralities for large graphs
approximately (currently betweenness and closeness):


#### Examples

1. Calculate betweenness centrality for [Pokec graph](
http://konect.uni-koblenz.de/networks/soc-pokec-relationships). The graph will be downloaded and 
giant component extracted.
```
python3 src/statistics.py -n Pokec -s BETWEENNESS_DISTR
```
