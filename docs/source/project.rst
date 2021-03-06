===================
Project description
===================

The CrawlingFramework (`source code <https://github.com/crawling-framework/crawling-framework.github.io>`_) is aimed for offline testing of network crawling algorithms on graph data. Undirected graphs without self-loops are supported yet.

**Features**:

#. Automatic graphs downloading from `networkrepository <http://networkrepository.com/>`_ online collections.
#. Graph statistics (including centralities) can be calculated and are stored together with the graphs.
#. Implement your own algorithm or use one of the already implemented from several families of crawlers:

   * popular ones: RandomCrawler, RandomWalkCrawler, BreadthFirstSearchCrawler, DepthFirstSearchCrawler, MaximumObservedDegreeCrawler, PreferentialObservedDegreeCrawler;
   * multicrawler mode - when a set of crawlers work together;
   * advanced crawlers - more complex strategies, e.g. described in [1], [2], and [3].
#. Run crawlers on given graphs, calculating quality measures, saving the history, and drawing comparison plots.

**Planning**:

* graph models: controlled assortativity, LFR, ERGG;
* extension for directed graphs.


Demo-tutorial
-------------

See 7-steps tutorial at :ref:`demo <demo-lbl>` for a quick start.

References
----------

[1] Avrachenkov, Konstantin, et al. "Quick detection of high-degree entities in large directed networks." 2014 IEEE International Conference on Data Mining. IEEE, 2014. `arxiv <https://arxiv.org/pdf/1410.0571.pdf>`_

[2] Madhawa, K., Murata, T. A multi-armed bandit approach for exploring partially observed networks. Appl Netw Sci 4, 26 (2019). `springer <https://doi.org/10.1007/s41109-019-0145-0>`_

[3] Three-step Algorithms for Detection of High Degree Nodes in Online Social Networks (2020) (TODO add ref when published) `link <>`_

