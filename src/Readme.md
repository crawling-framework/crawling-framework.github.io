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
* Install [SNAP](https://snap.stanford.edu/snap/index.html) C++ (in any directory):
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
* Install [SNAP](https://snap.stanford.edu/snap/index.html) C++ (in any directory):
```
git clone https://github.com/snap-stanford/snap.git
cd snap
make all
```

### Final preparations 

Copy the file `config.exmaple` to `config` - this file will contain your specific flags and 
paths.

#### SNAP C++ (for cythonic crawlers)

In the `config` file set variables
```
USE_CYTHON_CRAWLERS = True         # python/cython mode switcher
SNAP_DIR = "/path/to/snap"         # directory with snap built
```

Put there your path to the installed snap root directory.
NOTE: don't start the path from '~' or it will fail with the g++ option '-I'.

If you want to use slow pythonic crawlers, set `USE_CYTHON_CRAWLERS = False`.

The following steps are optional, they may be use to speed up computations.

#### Ligra (for fast eccentricity estimations)

Install [Ligra](https://github.com/jshun/ligra) framework to use approximate eccentricity
algorithm for large graphs.
```
git clone https://github.com/jshun/ligra.git
cd ligra/apps/eccentricity
export CILK=" "
make all
cd ../../utils
make all
```
Set corresponding variables in config file:
```
USE_LIGRA = True                     # Use Ligra library for approximate centrality calculation
LIGRA_DIR = "/path/to/ligra"         # directory with Ligra built
```

## Usage

One may toggle several switchers described above:
`USE_CYTHON_CRAWLERS` - to employ cythonic implementations,
`USE_NETWORKIT` - to use [Networkit](https://networkit.github.io/) library to compute 
centralities for large graphs approximately (currently betweenness and closeness), 
`USE_LIGRA` - to [Ligra](https://github.com/jshun/ligra) framework to use approximate eccentricity
algorithm for large graphs.


#### Examples

1. Calculate betweenness centrality for [Pokec graph](
http://konect.uni-koblenz.de/networks/soc-pokec-relationships). The graph will be downloaded and 
giant component extracted.
```
python3 src/statistics.py -n Pokec -c konect -s BETWEENNESS_DISTR
```
