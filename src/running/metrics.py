from numbers import Number
from time import time

from base.cgraph import MyGraph
from crawlers.cbasic import Crawler
from crawlers.declarable import Declarable, declaration_to_filename
from search.oracles import Oracle


class Metric(Declarable):
    """
    Base class for metrics on crawling process.
    Metric has a callback function that takes a crawler and returns some value.
    Metric filename is constructed only from class name and kwargs, so they should fully identify
    the metric.
    To create a custom metric, one should implement a subclass with proper callback().
    As examples see `OracleBasedMetric`.
    """
    def __init__(self, is_numeric, name=None, callback=None, **kwargs):
        """
        NOTE: is_numeric, name, and callback are not used in filename.

        :param is_numeric: if True, metric must return number, ResultsMerger can operate with it,
        the kwargs are used in folder name. If False, metric can return any structure which is
        saved in folder name as the class, ResultsMerger ignores it.
        :param name: name to use in plots
        :param callback: metric function: callback(crawler, **kwargs) -> number
        :param kwargs: additional argument to callback function
        """
        if is_numeric:  # Pass kwargs to be in folder name
            super(Metric, self).__init__(**kwargs)
        else:
            super(Metric, self).__init__(is_numeric=False)
        self._is_numeric = is_numeric
        self._callback = callback
        self._kwargs = kwargs
        self.name = name if name else declaration_to_filename(self._declaration)

    @property
    def is_numeric(self) -> bool:
        return self._is_numeric

    @staticmethod
    def from_declaration(declaration, **aux_kwargs):
        """ Build a Metric instance from its declaration """
        _class, _kwargs = declaration
        assert _class != Metric, "Create a subclass to define your own Metric"
        return _class(**_kwargs, **aux_kwargs)

    def __call__(self, crawler: Crawler):
        return self._callback(crawler, **self._kwargs)


class OracleBasedMetric(Metric):
    """ Outer metric based on target set that is determined by Oracle.
    Outer metric is some function of graph and common crawler properties (e.g. crawled/observed sets).
    """
    # short = 'OracleMetric'

    def __init__(self, graph: MyGraph, oracle: Oracle, measure='size', part='crawled', name=None):
        """ Compares crawling result to the target set.
        :param oracle: target node detector
        :param measure: 'Pr' (precision), 'Re' (recall), 'F1' (F1-score), or 'size' (just size)
        :param part: 'crawled', 'observed', 'nodes' (observed+crawled), 'answer' (crawler must
         support getting an answer, e.g. extend CrawlerWithAnswer)
        :param name: name for plotting
        """
        assert part in ['crawled', 'observed', 'nodes', 'answer']
        assert measure in ['Pr', 'Re', 'F1', 'size']

        get_part = {
            'crawled': lambda crawler: crawler.crawled_set,
            'observed': lambda crawler: crawler.observed_set,
            'nodes': lambda crawler: crawler.nodes_set,
            'answer': lambda crawler: crawler.answer,
        }[part]

        callback = {
            'Pr': lambda crawler, **kwargs:    len(get_part(crawler).intersection(oracle.target_set(graph))) / len(get_part(crawler)),
            'Re': lambda crawler, **kwargs:    len(get_part(crawler).intersection(oracle.target_set(graph))) / len(oracle.target_set(graph)),
            'F1': lambda crawler, **kwargs:  2*len(get_part(crawler).intersection(oracle.target_set(graph))) / (len(oracle.target_set(graph)) + len(get_part(crawler))),
            'size': lambda crawler, **kwargs:  len(get_part(crawler).intersection(oracle.target_set(graph))),
        }[measure]

        name = name if name else f"{measure} {part} {oracle.name}"
        super().__init__(True, name, callback, oracle=oracle, measure=measure, part=part)


class StatisticsSaverMetric(Metric):
    """
    StatisticsSaverMetric calls crawler.collect_statistics() result.
    If 'is_numeric' is True and some 'value_of' is specified, it extracts corresponding statistics
    value that is to be saved.

    NOTE: if the statistics is numeric if its value is number and it is measured at every batch.
    Statistics measured at every crawl cannot be numeric since they are flushed in batches and thus
    are vectors.
    Nevertheless, e.g. their values averaged over batch can be plotted via
    StatisticsResultsMerger.draw_function_of_metric_over_step() (see demo/demo_statistics.py).
    """
    short = "Statistics"

    def __init__(self, graph: MyGraph, is_numeric=False, value_of=None, name=None, **kwargs):
        """
        Measure crawling result with respect to top fraction of nodes by a specified centrality.

        :param is_numeric: If False, any data structure returned is saved as is, ResultsMerger will
        ignore it. If True, extract numeric value of a statistics given in 'value_of' and save it,
        ResultsMerger can operate with it.
        :param value_of: works only for is_numeric=True. Extracts numeric value of given statistics.
        :param name: name for plotting.
        :param kwargs: arguments for crawler.statistics(...), will be used in folder naming iff
        is_numeric==True.
        """
        if is_numeric:  # Extract numeric value
            def callback(crawler, **_kwargs):
                stat_dict = crawler.collect_statistics(**_kwargs)
                if value_of is None:
                    assert len(stat_dict) == 1
                    return next(iter(stat_dict.values()))
                val = stat_dict[value_of]
                if not isinstance(val, Number):
                    raise RuntimeError(
                        f"Statistics is declared numeric, but the returned value is not: {val}")
                return val
            name = value_of
            super().__init__(is_numeric, name, callback, value_of=value_of, **kwargs)

        else:  # Return as is
            def callback(crawler, **_kwargs):
                return crawler.collect_statistics(**_kwargs)

            super().__init__(is_numeric, name, callback, **kwargs)


class CallCounterMetric(Metric):
    """ Count number of calls of specific crawler method(s).
    NOTE: since metric is initialized after the first crawling step, this step is not counted,
    therefore the metric gives 1 fewer methods calls (e.g. for 'crawl', 'next_seed').
    """
    short = "Calls"

    def __init__(self, graph: MyGraph, methods: (str, list), name=None):
        self.methods = methods
        if isinstance(methods, str):
            is_numeric = True
            self.methods = [methods]
        else:
            is_numeric = False

        self.patched_crawlers = set()
        self.crawler_method_count = {}  # crawler -> method -> count

        def callback(crawler, **kwargs):
            self.patch_crawler(crawler)
            if is_numeric:
                return self.crawler_method_count[crawler][self.methods[0]]
            return self.crawler_method_count[crawler].copy()

        name = name or f"Calls of '{methods}'"
        super().__init__(is_numeric, name, callback, methods=methods)

    class patched_method:
        """ Function which replaces original crawler method. """
        def __init__(self, method_name, method, counter):
            self.method_name = method_name
            self.method = method
            self.counter = counter

        def __call__(self, *args, **kwargs):
            self.counter[self.method_name] += 1
            return self.method(*args, **kwargs)

    def patch_crawler(self, crawler):
        """ Patch crawler methods if not yet. """
        if crawler in self.patched_crawlers:
            return

        # Patch crawler methods
        self.crawler_method_count[crawler] = {m: 0 for m in self.methods}
        for method_name in self.methods:
            method = getattr(crawler, method_name)
            setattr(crawler, method_name,
                    self.patched_method(method_name, method, self.crawler_method_count[crawler]))
        self.patched_crawlers.add(crawler)


class MethodTimerMetric(Metric):
    """ Measure the time spent by specific crawler method(s).
    """
    short = "Timer"

    def __init__(self, graph: MyGraph, methods: (str, list), name=None):
        self.methods = methods
        if isinstance(methods, str):
            is_numeric = True
            self.methods = [methods]
        else:
            is_numeric = False

        self.patched_crawlers = set()
        self.crawler_method_seconds = {}

        def callback(crawler, **kwargs):
            self.patch_crawler(crawler)
            if is_numeric:
                return self.crawler_method_seconds[crawler][self.methods[0]]
            return self.crawler_method_seconds[crawler].copy()

        name = name or f"Time of '{methods}', s"
        super().__init__(is_numeric, name, callback, methods=methods)

    class patched_method:
        """ Function which replaces original crawler method. """
        def __init__(self, method_name, method, counter):
            self.method_name = method_name
            self.method = method
            self.counter = counter

        def __call__(self, *args, **kwargs):
            t = time()
            res = self.method(*args, **kwargs)
            self.counter[self.method_name] += time() - t
            return res

    def patch_crawler(self, crawler):
        """ Patch crawler methods if not yet. """
        if crawler in self.patched_crawlers:
            return

        # Patch crawler methods
        self.crawler_method_seconds[crawler] = {m: 0 for m in self.methods}
        for method_name in self.methods:
            method = getattr(crawler, method_name)
            setattr(crawler, method_name,
                    self.patched_method(method_name, method, self.crawler_method_seconds[crawler]))
        self.patched_crawlers.add(crawler)


class NeighborhoodMetric(Metric):
    """ Custom numeric function of an observed node neighborhood.
    Examples of functions:
    >>> # Neighborhood -> observed degree
    >>> def neigh_od(neighborhood: Neighborhood):
    >>>     return neighborhood.od
    """
    short = "Neighborhood"

    def __init__(self, graph, function, function_name, name=None):
        """
        :param function: Neighborhood -> number
        :param function_name: obligate function name
        """
        is_numeric = True
        self.function = function

        self.patched_crawlers = set()
        self.crawler_counters = {}
        self.crawler_values = {}

        def callback(crawler, **kwargs):
            self.patch_crawler(crawler)
            # Zero counters on call, i.e. value is averaged over batch
            avg = self.crawler_values[crawler] / self.crawler_counters[crawler] \
                if self.crawler_counters[crawler] > 0 else 0
            self.crawler_counters[crawler] = 0
            self.crawler_values[crawler] = 0
            return avg

        name = name or f"{function_name}(neighborhood)"
        super().__init__(is_numeric, name, callback, function_name=function_name)

    class patched_method:
        """ Method which replaces original crawler method. """
        def __init__(self, crawler, method, crawler_counters, crawler_values, function):
            self.crawler = crawler
            self.method = method
            self.crawler_counters = crawler_counters
            self.crawler_values = crawler_values
            self.function = function

        def __call__(self, *args, **kwargs):
            # Call original method
            neighborhood = self.method(*args, **kwargs)

            # Apply measuring function
            self.crawler_counters[self.crawler] += 1
            self.crawler_values[self.crawler] += self.function(neighborhood)

            # Return original method result
            return neighborhood

    def patch_crawler(self, crawler):
        """ Patch crawler methods if not yet. """
        if crawler in self.patched_crawlers:
            return
        # Patch crawler methods
        self.crawler_counters[crawler] = 0
        self.crawler_values[crawler] = 0
        method = crawler.get_neighborhood
        crawler.get_neighborhood = self.patched_method(
            crawler, method, self.crawler_counters, self.crawler_values, self.function)
        self.patched_crawlers.add(crawler)

