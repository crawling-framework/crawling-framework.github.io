.. CrawlingFramework documentation master file, created by
   sphinx-quickstart on Tue Sep  1 15:35:11 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Code Documentation
******************
.. toctree::
   :maxdepth: 2
   :caption: Contents:

basics
===================
.. automodule:: base.cgraph
   :members: 
   :special-members: __init__
.. automodule:: graph_io
   :members: GraphCollections, temp_dir
   :special-members: __init__
.. automodule:: graph_stats
   :members:


crawlers
===================

.. automodule:: crawlers.cbasic
   :members: definition_to_filename, filename_to_definition

classic
-----------
.. automodule:: crawlers.cbasic
   :members: Crawler, RandomCrawler, RandomWalkCrawler, BreadthFirstSearchCrawler, DepthFirstSearchCrawler, SnowBallCrawler, MaximumObservedDegreeCrawler, PreferentialObservedDegreeCrawler, MaximumExcessDegreeCrawler
   :special-members: __init__
..   :exclude-members: definition_to_filename, filename_to_definition


advanced
-----------
.. automodule:: crawlers.advanced
   :members: CrawlerWithAnswer, AvrachenkovCrawler, ThreeStageCrawler, ThreeStageMODCrawler
   :special-members: __init__
.. automodule:: crawlers.cadvanced
   :members: DE_Crawler
   :special-members: __init__


ml-crawlers
-----------
.. automodule:: crawlers.ml.with_features
   :members:
   :special-members: __init__
.. automodule:: crawlers.ml.regression_reward
   :members:
   :special-members: __init__
.. automodule:: crawlers.ml.knn_ucb
   :members:
   :special-members: __init__


runners
===================
.. automodule:: running.metrics_and_runner
   :members:
   :special-members: __init__
.. automodule:: running.animated_runner
   :members:
   :special-members: __init__
.. automodule:: running.visual_runner
   :members:
   :special-members: __init__
.. automodule:: running.history_runner
   :members: CrawlerHistoryRunner
   :special-members: __init__
.. automodule:: running.merger
   :members:
   :special-members: __init__


demo
===================
.. _demo-lbl:
.. automodule:: demo.demo
   :members:

advanced usage
-------------------

For a more advanced usage see examples in the code.

To create a **custom crawler** you should extend the root class Crawler from *base.crawlers.cbasic.py*. See examples in *base.crawlers.advanced.py*. For more complex algorithms using machine learning see *base.crawlers.ml*. To make it faster you could also implement your crawler in cython, see examples in *base.crawlers.advanced.pyx*.

To create a **custom metric** take a look at *running.metrics_and_runner.py*. You should extend the root class Metric for that.


