import logging

from base.cgraph import MyGraph
from crawlers.cbasic import Crawler, NoNextSeedError,\
    MaximumObservedDegreeCrawler, PreferentialObservedDegreeCrawler, definition_to_filename

logger = logging.getLogger(__name__)


class MultiCrawler(Crawler):
    """ General class for complex crawling strategies using several crawlers
    """
    pass


class MultiInstanceCrawler(MultiCrawler):
    """
    Runs several crawlers in parallel. Each crawler makes a step iteratively in a cycle.
    When the crawler can't get next seed it is discarded from the cycle.
    """
    short = 'MultiInstance'
    def __init__(self, graph: MyGraph, name: str=None, count: int=0, crawler_def=None):
        """
        :param graph:
        :param count: how many instances to use
        :param crawler_def: crawler instance definition as (class, kwargs)
        """
        assert crawler_def is not None
        assert count < graph.nodes()

        super().__init__(graph, name=name, count=count, crawler_def=crawler_def)

        self.crawler_def = crawler_def  # FIXME can we speed it up?
        self.crawlers = []
        self.keep_node_owners = False  # True if any crawler is MOD or POD
        self.node_owner = {}  # node -> index of crawler who owns it. Need for MOD, POD

        seeds = graph.random_nodes(count)
        [self.observe(s) for s in seeds]

        _class, kwargs = crawler_def

        # Create crawler instances and init them with different random seeds
        for i in range(count):
            crawler = _class(graph, initial_seed=seeds[i],
                             observed_graph=self._observed_graph, crawled_set=self._crawled_set,
                             observed_set={seeds[i]})
            self.crawlers.append(crawler)

            if isinstance(crawler, MaximumObservedDegreeCrawler) or isinstance(crawler, PreferentialObservedDegreeCrawler):
                n = seeds[i]
                self.node_owner[n] = crawler
                self.keep_node_owners = True

        if not name:
            self.name = 'Multi%s%s' % (count, self.crawlers[0].name)  # short name for pics
        self.next_crawler = 0  # next crawler index to run

    def crawl(self, seed: int) -> list:
        """ Run the next crawler.
        """
        c = self.crawlers[self.next_crawler]  # FIXME ref better?
        res = c.crawl(seed)
        logger.debug("res of crawler[%s]: %s" % (self.next_crawler, [n for n in res]))

        assert seed in self._crawled_set  # FIXME do we need it?
        assert seed in self._observed_set  # FIXME potentially error if node was already removed
        self._observed_set.remove(seed)  # removed crawled node
        for n in res:
            self._observed_set.add(n)  # add newly observed nodes

        if self.keep_node_owners:  # TODO can we speed it up?
            # update owners dict
            del self.node_owner[seed]
            for n in res:
                self.node_owner[n] = self.crawlers[self.next_crawler]

            # distribute nodes with changed degree among instances to update their priority structures
            for n in self._observed_graph.neighbors(seed):
                if n in self.node_owner:
                    c = self.node_owner[n]
                    if c != self.crawlers[self.next_crawler] and (isinstance(c, MaximumObservedDegreeCrawler) or isinstance(c, PreferentialObservedDegreeCrawler)):
                        c.update([n])

        self.next_crawler = (self.next_crawler+1) % len(self.crawlers)
        # self.seed_sequence_.append(seed)
        return res

    def next_seed(self) -> int:
        """ The next crawler makes a step. If impossible, it is discarded.
        """
        for _ in range(len(self.crawlers)):
            try:
                s = self.crawlers[self.next_crawler].next_seed()
            except NoNextSeedError as e:
                logger.debug("Run crawler[%s]: %s Removing it." % (self.next_crawler, e))
                # print("Run crawler[%s]: %s Removing it." % (self.next_crawler, e))
                del self.crawlers[self.next_crawler]
                # idea - create a new instance
                self.next_crawler = self.next_crawler % len(self.crawlers)
                continue

            logger.debug("Crawler[%s] next seed=%s" % (self.next_crawler, s))
            # print("Crawler[%s] next seed=%s" % (self.next_crawler, s))
            return s

        raise NoNextSeedError("None of %s subcrawlers can get next seed." % len(self.crawlers))
