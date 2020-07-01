import datetime
import glob
import json
import multiprocessing
import os
import logging
from tqdm import tqdm
import multiprocessing as mp
import traceback
from time import time

from crawlers.advanced import ThreeStageCrawler, ThreeStageMODCrawler

from base.cgraph import MyGraph, seed_random
from crawlers.cbasic import Crawler, definition_to_filename, filename_to_definition
from graph_io import GraphCollections
from running.metrics_and_runner import CrawlerRunner, TopCentralityMetric, Metric
from running.merger import ResultsMerger
from statistics import Stat


def send_misha_vk(msg: str):
    """ Try to send message to misha's VK """
    try:
        from utils import rel_dir
        bot_path = os.path.join(rel_dir, "src", "experiments", "vk_signal.py")
        os.system("python3 %s -m '%s'" % (bot_path, msg))
    except: pass


class Process(mp.Process):
    """ multiprocessing.Process which handles exceptions
    from https://stackoverflow.com/a/33599967/8900030
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
            send_misha_vk(self._pconn.recv()[1])
            raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


class CrawlerHistoryRunner(CrawlerRunner):
    """ Runs crawlers, measure metrics and save history. Step sequence is logarithmically growing,
    universal for all graphs
    """
    def __init__(self, graph: MyGraph, crawler_defs, metric_defs, budget: int = -1, step: int = -1):
        """
        :param graph: graph to run
        :param crawler_defs: list of crawler definitions to run. Crawler definitions will be
         initialized when run() is called
        :param metric_defs: list of metric definitions to compute at each step. Metric should be
         callable function crawler -> float, and have name
        :param budget: maximal number of nodes to be crawled, by default the whole graph
        :param step: compute metrics each `step` steps
        :return:
        """
        super().__init__(graph, crawler_defs=crawler_defs, metric_defs=metric_defs, budget=budget, step=step)

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

    def run(self, same_initial_seed=False):
        """ Run crawlers and measure metrics. In the end, the measurements are saved.

        :param graph: graph to run
        :param same_initial_seed: use the same initial seed for all crawler instances
        :param draw_networkx: if True draw graphs with colored nodes using networkx and save it as
         gif. Use for small graphs only.
        :param budget: maximal number of nodes to be crawled, by default the whole graph
        :param step: compute metrics each `step` steps
        :return:
        """
        crawlers, metrics, batch_generator = self._init_runner(same_initial_seed)
        pbar = tqdm(total=self.budget, desc='Running iterations')  # drawing crawling progress bar

        if int(10000*time()) % 2 == 0:
            raise Exception("test exception")

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
        self._save_history(crawler_metric_seq, step_seq)  # saving ending history
        logging.info("Finished running at %s" % (datetime.datetime.now()))

    def run_parallel(self, num_processes=multiprocessing.cpu_count()):
        """
        Run in parallel crawlers and measure metrics. In the end, the measurements are saved.

        :param num_processes: number of processes to run simultaneously
        :return:
        """
        t = time()
        jobs = []
        for i in range(num_processes):
            logging.info('Start parallel job %s of %s' % (i+1, num_processes))
            seed_random(int(time() * 1e7 % 1e9))  # to randomize t_random in parallel processes
            p = Process(target=self.run)
            jobs.append(p)
            p.start()
            # FIXME it could be a problem when several running try to read and write into one directory

        errors = 0
        for i, p in enumerate(jobs):
            p.join()
            if p.exception:
                errors += 1
            logging.info('Complete parallel job %s of %s' % (i+1, num_processes))

        msg = 'Completed %s of %s runs for graph %s with N=%s E=%s. Time elapsed %.1fs.' % (
            num_processes - errors, num_processes,
            self.graph.name, self.graph.nodes(), self.graph.edges(), time() - t)

        # print(msg)
        return msg

    def run_parallel_adaptive(self, n_instances: int = 10,
                              max_cpus: int = multiprocessing.cpu_count(), max_memory: float = 6):
        """
        Runs in parallel crawlers and measure metrics. Number of processes is chosen adaptively.
        Using magic coefficients: Mbytes of memory = A*n + B,
        where A = 0.25, B = 2.5,  n - thousands of nodes in graph.

        :param n_instances: total wanted number of instances to be performed
        :param max_cpus: max number of CPUs to use for computation, all available by default
        :param max_memory: max Mbytes of operative memory to use for computation, 6Gb by default
        :return:
        """
        # Gbytes of operative memory per instance
        memory = (0.25 * self.graph['NODES'] / 1000 + 2.5) / 1024 * len(self.crawler_defs)
        max_cpus = min(max_cpus, int(max_memory // memory))

        while n_instances > 0:
            num = min(max_cpus, n_instances)
            n_instances -= num
            msg = self.run_parallel(num_processes=num)

            send_misha_vk(msg)

    def run_missing(self, n_instances=10,
                    max_cpus: int = multiprocessing.cpu_count(), max_memory: float = 6):
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
            logging.info("No missing experiments found.")
            return

        cmi = missing[self.graph.name]
        self.crawler_defs = [filename_to_definition(c) for c in cmi.keys()]  # only missing ones
        max_count = 0
        for crawler_name, mi in cmi.items():
            max_count = max(max_count, max(mi.values()))
        # TODO maybe choose missing cases in more details to avoid redundant runs?

        logging.info("Will run %s missing iterations for %s crawlers, %s metrics on graph %s." % (
            max_count, len(cmi.keys()), len(self.metric_defs), self.graph.name))

        # Parallel run with adaptive number of CPUs
        self.run_parallel_adaptive(n_instances=max_count, max_cpus=max_cpus, max_memory=max_memory)


def test_history_runner():
    g = GraphCollections.get('digg-friends')

    p = 0.01
    budget = int(0.005 * g.nodes())
    s = int(budget / 2)

    crawler_defs = [
        (ThreeStageCrawler, {'s': s, 'n': budget, 'p': p}),
        (ThreeStageMODCrawler, {'s': s, 'n': budget, 'p': p}),
    ]
    metric_defs = [
        (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'nodes'}),
        (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'Re', 'part': 'answer'}),
        # (TopCentralityMetric, {'top': p, 'centrality': Stat.DEGREE_DISTR.short, 'measure': 'F1', 'part': 'answer'}),
    ]
    n_instances = 5
    # Run missing iterations
    chr = CrawlerHistoryRunner(g, crawler_defs, metric_defs)
    # chr.run_missing(n_instances, max_cpus=4, max_memory=2.5)
    chr.run_parallel(2)
    # chr.run_parallel_adaptive(4, max_cpus=2, max_memory=2.5)

    # # Run merger
    # crm = ResultsMerger([g.name], crawler_defs, metric_defs, n_instances)
    # crm.draw_by_metric_crawler(x_lims=(0, budget), x_normalize=False, scale=8, swap_coloring_scheme=True, draw_error=False)


if __name__ == '__main__':
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    test_history_runner()
