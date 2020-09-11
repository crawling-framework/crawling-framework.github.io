# Network crawling framework

_description_

**Features**:
* Automatic graphs downloading from [Konect](http://konect.cc/networks/) and 
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
* cmake

For MacOS additional:

* brew

## Installation and usage

See the docs.