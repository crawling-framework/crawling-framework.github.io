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
======
.. automodule:: crawlers.declarable
   :members: Declarable, declaration_to_filename, filename_to_declaration
.. automodule:: base.cgraph
   :members: 
   :special-members: __init__
.. automodule:: graph_io
   :members: GraphCollections
   :special-members: __init__


predictors
==========

.. automodule:: search.predictors.simple_predictors
   :members: Predictor, MaximumTargetNeighborsPredictor, SklearnPredictor

.. automodule:: search.predictors.gnn_predictors
   :members: GNNet, GNNPredictor


crawlers
========

simple
------
.. automodule:: crawlers.cbasic
   :members: Crawler, InitialSeedCrawlerHelper, RandomCrawler, RandomWalkCrawler, BreadthFirstSearchCrawler, DepthFirstSearchCrawler, SnowBallCrawler, MaximumObservedDegreeCrawler, PreferentialObservedDegreeCrawler, MaximumExcessDegreeCrawler, MaximumRealDegreeCrawler
   :special-members: __init__
..   :exclude-members: definition_to_filename, filename_to_definition


single-predictor
----------------
.. automodule:: search.predictor_based_crawlers.predictor_based
   :members: PredictorBasedCrawler
   :special-members: __init__

multi-predictor
---------------
.. automodule:: search.predictor_based_crawlers.mab
   :members: MultiPredictorCrawler, MABCrawler, ExponentialDynamicWeightsMultiPredictorCrawler, FollowLeaderMABCrawler, BetaDistributionMultiPredictorCrawler
   :special-members: __init__


running
=======
.. automodule:: running.history_runner
   :members: SmartCrawlersRunner
.. automodule:: running.merger
   :members: ResultsMerger
   :special-members: __init__
.. automodule:: running.metrics
   :members: Metric
   :special-members: __init__

