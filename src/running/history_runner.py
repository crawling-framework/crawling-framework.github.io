import datetime
import json
import logging
import multiprocessing as mp
import os
import traceback
from math import ceil
from time import time, sleep

from base.cgraph import MyGraph, seed_random
from crawlers.cadvanced import DE_Crawler
from crawlers.cbasic import definition_to_filename, filename_to_definition, \
    RandomWalkCrawler, RandomCrawler, BreadthFirstSearchCrawler, DepthFirstSearchCrawler, \
    SnowBallCrawler, MaximumObservedDegreeCrawler
from tqdm import tqdm

from crawlers.multiseed import MultiInstanceCrawler
from graph_io import GraphCollections
from graph_stats import Stat
from running.merger import ResultsMerger
from running.metrics_and_runner import CrawlerRunner, TopCentralityMetric, centrality_by_name


def send_vk(msg: str):
    """ Try to send message to VK account """
    try:
        from utils import rel_dir, VK_ID
        if VK_ID != "00000000":  # default value from config.example which means to not send
            bot_path = os.path.join(rel_dir, "src", "experiments", "vk_signal.py")
            os.system("python3 %s -m '%s' --id '%s'" % (bot_path, msg, VK_ID))
    except:
        pass


class Process(mp.Process):
    """ multiprocessing.Process that handles exceptions.
    From https://stackoverflow.com/a/33599967/8900030
    """
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            send_vk(self._pconn.recv()[1])
            raise e

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class CrawlerHistoryRunner(CrawlerRunner):
    """
    Runs several crawlers and measures several metrics for a given graph.
    Saves measurements history to the disk.
    Can run several instances of a crawler in parallel.
    Step sequence is exponentially growing independent of the graph.

    Functions:
    ----------
    * run - Run given crawlers and measure metrics. In the end, the measurements are saved to files.
    * run_parallel - Run the configuration  using the specified number of processes.
    * run_parallel_adaptive - Runs parallel computations adaptively. The configuration is split such
      that not to exceed specified memory limits.
    * run_missing - Find missing configurations and run experiments adaptively. Convenient if some
      of previous experiments failed.

    NOTES:
    ------
    * step sequence when metrics are computed must be the same for all runs. Otherwise the results
      would not be merged correctly

    """

    def __init__(self, graph: MyGraph, crawler_defs, metric_defs, budget: int = -1, step: int = -1):
        """
        :param graph: graph to run
        :param crawler_defs: list of crawler definitions to run. Crawler definitions will be
         initialized when run() is called
        :param metric_defs: list of metric definitions to compute at each step. Metric should be
         callable function crawler -> float, and have name
        :param budget: maximal number of nodes to be crawled, by default the whole graph
        :param step: compute metrics each `step` steps, by default exponential step
        :return:
        """
        super().__init__(graph, crawler_defs=crawler_defs, metric_defs=metric_defs, budget=budget, step=step)
        self._init_semaphore = mp.Semaphore(1)
        self._save_semaphore = mp.Semaphore(1)

    def _save_history(self, crawler_metric_seq, step_seq):
        pbar = tqdm(total=len(crawler_metric_seq), desc='Saving history')
        for crawler in crawler_metric_seq.keys():
            metric_seq = crawler_metric_seq[crawler]
            for metric in metric_seq.keys():
                metric_value = dict((step_seq[it], value) for it, value in enumerate(metric_seq[metric]))
                path_pattern = ResultsMerger.names_to_path(
                    self.graph.name, definition_to_filename(crawler.definition), definition_to_filename(metric.definition))
                directory = os.path.dirname(path_pattern)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                ix = 0
                while True:
                    path = path_pattern.replace('*', str(ix))
                    if not os.path.exists(path):  # if name exists, adding number to it
                        break
                    ix += 1

                with open(path, 'w') as f:
                    json.dump(metric_value, f, indent=1)
            pbar.update(1)
        pbar.close()

    def run(self):
        """ Run crawlers and measure metrics. In the end, the measurements are saved.
        Note, this function can be called in parallel processes.
        """
        with self._init_semaphore:
            # To ensure stats reading/calculation only once in case of parallel run
            crawlers, metrics, batch_generator = self._init_runner()

        pbar = tqdm(total=self.budget, desc='Running iterations')  # drawing crawling progress bar

        step = 0
        step_seq = [0]  # starting for point 0
        crawler_metric_seq = dict([(c, dict([(m, [0]) for m in metrics])) for c in crawlers])
        for batch in batch_generator:
            step += batch
            step_seq.append(step)

            for crawler in crawlers:
                crawler.crawl_budget(batch)
                for metric in metrics:
                    # Calculate metric for crawler
                    value = metric(crawler)
                    crawler_metric_seq[crawler][metric].append(value)

            pbar.update(batch)

        pbar.close()  # closing progress bar

        with self._save_semaphore:
            # When saving history, it needs to know the number of files in the directory
            self._save_history(crawler_metric_seq, step_seq)

        logging.info("Finished running at %s" % (datetime.datetime.now()))

    def run_parallel(self, num_processes=mp.cpu_count()):
        """
        Run in parallel crawlers and measure metrics. In the end, the measurements are saved.

        :param num_processes: number of processes to run simultaneously
        :return:
        """
        t = time()

        # Some optimization hacks.
        # We load graph if it's not yet, to avoid doing it in each process
        if not self.graph.is_loaded():
            self.graph.load()

        # We try to load graph stats from file to avoid doing it in each process
        for md in self.metric_defs:
            _, kwargs = md
            if 'centrality' in kwargs:
                s = self.graph[centrality_by_name[kwargs['centrality']]]

        jobs = []
        for i in range(num_processes):
            logging.info('Start parallel job %s of %s' % (i+1, num_processes))
            seed_random(int(time() * 1e7 % 1e9))  # to randomize t_random in parallel processes
            p = Process(target=self.run)
            jobs.append(p)
            p.start()

        errors = 0
        for i, p in enumerate(jobs):
            p.join()
            err_msg = ''
            if p.exception:
                errors += 1
                err_msg = ' with exception: %s' % p.exception
            logging.info('Complete parallel job %s of %s%s' % (i+1, num_processes, err_msg))

        msg = 'Completed %s of %s runs for graph %s with N=%s E=%s. Time elapsed %.1fs.' % (
            num_processes - errors, num_processes,
            self.graph.name, self.graph.nodes(), self.graph.edges(), time() - t)

        return msg

    def run_parallel_adaptive(self, n_instances: int = 10,
                              max_cpus: int = mp.cpu_count(), max_memory: float = 4):
        """
        Runs in parallel crawlers and measure metrics. Number of processes is chosen adaptively.
        Using magic coefficients: Mbytes of memory = A*n + B*e + C,
        where A = 0.25, B = 0.01, C = 2.5,  n - thousands of nodes in graph, e - thousands of edges.

        :param n_instances: total wanted number of instances to be performed
        :param max_cpus: max number of CPUs to use for computation, all available by default
        :param max_memory: max Gbs of operative memory to use for computation, 6 Gb by default
        :return:
        """
        # GBytes of operative memory per 1 crawler
        m = (0.25 * self.graph[Stat.NODES] / 1000 + 0.01 * self.graph[Stat.EDGES] / 1000 + 2.5) / 1024
        if m > max_memory:
            raise MemoryError(
                "Not enough memory to even run 1 configuration: %s < %s needed" % (max_memory, m))

        if len(self.crawler_defs) > n_instances:  # split by crawler_defs
            # Prefer to take max possible instances
            max_cpus = min(max_cpus, int(max_memory // m))
            while n_instances > 0:
                num = min(max_cpus, n_instances)
                n_instances -= num

                # Now split crawler_def into parts if needed
                memory = m * num
                max_cds = int(max_memory // memory)
                assert max_cds >= 1
                crawler_defs = self.crawler_defs
                left = len(crawler_defs)
                logging.info("Split %s crawler_defs into %s parts" % (left, ceil(left / max_cds)))
                last_ix = 0
                while left > 0:
                    batch = min(max_cds, left)
                    self.crawler_defs = crawler_defs[last_ix: last_ix+batch]
                    last_ix += batch
                    left -= batch
                    msg = self.run_parallel(num_processes=num) + ", %s of %s crawler_defs left" % (left, len(crawler_defs))
                    send_vk(msg)

        else:  # split by instances
            memory = m * len(self.crawler_defs)
            max_cpus = min(max_cpus, int(max_memory // memory))
            assert max_cpus >= 1

            while n_instances > 0:
                num = min(max_cpus, n_instances)
                n_instances -= num
                msg = self.run_parallel(num_processes=num)

                send_vk(msg)

    def run_missing(self, n_instances=10,
                    max_cpus: int = mp.cpu_count(), max_memory: float = 6):
        """
        Runs all missing experiments for the graph. All crawlers and metrics run simultaneously, the
        number of instances is maximal among missing ones.

        :param n_instances: minimal wanted number of instances
        :param max_cpus: max number of CPUs to use for computation, all by default
        :param max_memory: max Mbytes of operative memory to use for computation, 6Gb by default
        :return:
        """
        # Get missing combinations
        crm = ResultsMerger([self.graph.name], self.crawler_defs, self.metric_defs, n_instances=n_instances)
        missing = crm.missing_instances()

        if len(missing) == 0:
            logging.info("No missing experiments found for graph '%s'." % self.graph.name)
            return

        cmi = missing[self.graph.name]
        self.crawler_defs = [filename_to_definition(c) for c in cmi.keys()]  # only missing ones
        max_count = 0
        for crawler_name, mi in cmi.items():
            max_count = max(max_count, max(mi.values()))
        # TODO choose missing cases more detailed to avoid redundant runs

        logging.info("Will run %s missing iterations for %s crawlers, %s metrics on graph %s." % (
            max_count, len(cmi.keys()), len(self.metric_defs), self.graph.name))

        # Parallel run with adaptive number of CPUs
        self.run_parallel_adaptive(n_instances=max_count, max_cpus=max_cpus, max_memory=max_memory)


def test_history_runner():
    g = GraphCollections.get('petster-hamster')
    # g = GraphCollections.get('Pokec')

    p = 0.01

    crawler_defs = [
        # (ThreeStageCrawler, {'s': s, 'n': budget, 'p': p}),
        # (ThreeStageMODCrawler, {'s': s, 'n': budget, 'p': p}),
        (RandomWalkCrawler, {}),
        (RandomCrawler, {}),
        (BreadthFirstSearchCrawler, {}),
        (DepthFirstSearchCrawler, {}),
        (SnowBallCrawler, {'p': 0.1}),
        (MaximumObservedDegreeCrawler, {'batch': 1}),
        (MaximumObservedDegreeCrawler, {'batch': 10}),
        (DE_Crawler, {}),
        (MultiInstanceCrawler, {'count': 5, 'crawler_def': (MaximumObservedDegreeCrawler, {})}),
    ]
    metric_defs = [
        (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'}),
        # (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'answer'}),
        # (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'F1', 'part': 'answer'}),
    ]
    n_instances = 5
    # Run missing iterations
    chr = CrawlerHistoryRunner(g, crawler_defs, metric_defs)
    # chr.run_missing(n_instances, max_cpus=4, max_memory=2.5)
    chr.run_parallel(n_instances)
    # chr.run_parallel_adaptive(4, max_cpus=2, max_memory=2.5)

    # # Run merger
    # crm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)
    # crm.draw_by_metric_crawler(x_lims=(0, budget), x_normalize=False, scale=8, swap_coloring_scheme=True, draw_error=False)


if __name__ == '__main__':
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    test_history_runner()
