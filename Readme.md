# Network crawling framework

The CrawlingFramework is aimed for offline testing of network crawling algorithms on graph data. 
Undirected graphs without self-loops are supported yet.

**Features**:
* Automatic graphs downloading from [networkrepository](http://networkrepository.com/) online 
collection.
* Graph statistics (including centralities) can be calculated and are stored together with the
 graphs.
* Implement your own algorithm or use one of the already implemented from several families of 
crawlers:
  * popular ones: RandomCrawler, RandomWalkCrawler, BreadthFirstSearchCrawler, 
  DepthFirstSearchCrawler, MaximumObservedDegreeCrawler, PreferentialObservedDegreeCrawler;
  * multicrawler mode - when a set of crawlers work together;
  * advanced crawlers - for more complex strategies.
* Run crawlers on given graphs, calculating quality measures, saving the history, and drawing 
plots.

**Planning**:
* graph models: controlled assortativity, ERGG.

## Installation and usage

See the documentation and tutorials at https://crawling-framework.github.io/
