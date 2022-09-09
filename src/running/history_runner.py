import datetime
import json
import logging
import multiprocessing as mp
import os

from base.cgraph import MyGraph
from tqdm import tqdm

from crawlers.cbasic import Crawler
from crawlers.declarable import declaration_to_filename, filename_to_declaration
from graph_io import GraphCollections
from running.knapsack.jobs_balancer import GreedyBalancer
from running.knapsack.jobs_runner import JobProcess, JobsRunner
from running.merger import ResultsMerger
from running.metrics import Metric
from running.runner import CrawlerRunner


class SmartCrawlersRunner:
    """
    Runs several crawlers and measures several metrics for a given graph.
    Saves measurements history to the disk.
    Can run several instances of a crawler in parallel.
    Step sequence is exponentially growing independent of the graph.
    """
    def __init__(self, graph_full_names, crawler_decls, metric_decls, budget: int = -1):
        self.graph_full_names = graph_full_names
        self.crawler_decls = crawler_decls
        self.metric_decls = metric_decls
        self.budget = budget

    def run_adaptive(self, n_instances=10, max_cpus: int=mp.cpu_count(),
                     max_memory: float=0.7 * os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3)):
        fn_decl = {}  # filename -> original crawler/metric declaration
        for cd in self.crawler_decls:
            c = Crawler.from_declaration(cd, graph=None)
            f = declaration_to_filename(c.declaration)
            fn_decl[f] = cd
        for md in self.metric_decls:
            m = Metric.from_declaration(md, graph=None)
            f = declaration_to_filename(m.declaration)
            fn_decl[f] = md

        # Get missing combinations
        crm = ResultsMerger(self.graph_full_names, self.crawler_decls, self.metric_decls,
                            self.budget, n_instances=n_instances)
        missing = crm.missing_instances()  # g -> c -> m -> count

        if len(missing) == 0:
            logging.info("No missing experiments found")
            return

        print(f"Missing configs to be run:")
        total_runs = 0
        for g, cmc in missing.items():
            for cd, mc in cmc.items():
                count = max(mc.values())
                ms = list(mc.keys())
                print(f"[{count}] {g}, {cd}, {ms}")
                total_runs += count
                cmc[cd] = (ms, count)
        # Now g -> c -> ([metrics], count)

        # Create jobs
        def run(graph_full_name=None, crawler_fn=None, metric_fns=None, **kwargs):
            g = GraphCollections.get(*graph_full_name)
            mds = [fn_decl[m] for m in metric_fns]
            cds = [fn_decl[crawler_fn]]
            chr = CrawlerHistoryRunner(g, cds, mds, self.budget)
            chr.run()

        jobs = []
        for g, cmc in missing.items():
            for crawler, (metrics, count) in cmc.items():
                cpu = mp.cpu_count() / 2 if 'GNNPredictor' in crawler else 1
                ram = 1400  # TODO estimate!
                # job_id = ','.join([str(g), c, str(ms)])
                job_id = None
                for _ in range(count):
                    jobs.append(JobProcess(
                        cpu=cpu, ram=ram, id=job_id, target=run,
                        kwargs={
                            'graph_full_name': g, 'crawler_fn': crawler, 'metric_fns': metrics,
                            'budget': self.budget}))

        runner = JobsRunner(max_cpus, max_memory * 1024, GreedyBalancer)
        runner.run(jobs)
        print(runner.history)


class CrawlerHistoryRunner(CrawlerRunner):
    """
    Runs several crawlers and measures several metrics for a given graph.
    Saves measurements history to the disk.
    Can run several instances of a crawler in parallel.
    Step sequence is exponentially growing independent of the graph.

    Functions:
    ----------
    * run - Run given crawlers and measure metrics. In the end, the measurements are saved to files.

    NOTES:
    ------
    * step sequence when metrics are computed must be the same for all runs. Otherwise the results
      would not be merged correctly

    """

    def __init__(self, graph: MyGraph, crawler_decls, metric_decls, budget: int = -1,
                 step: int = -1, omit_nones=True):
        """
        :param graph: graph to run
        :param crawler_decls: list of crawler declarations to run. Crawler declarations will be
         initialized when run() is called
        :param metric_decls: list of metric declarations to compute at each step. Metric should be
         callable function crawler -> float, and have name
        :param budget: maximal number of nodes to be crawled, by default the whole graph
        :param step: compute metrics each `step` steps, by default exponential step
        :param omit_nones: If False, metric values where all values are None will not be saved.
        By default save all metric values.
        """
        super().__init__(graph, crawler_decls=crawler_decls, metric_decls=metric_decls,
                         budget=budget, step=step)
        self._init_semaphore = mp.Semaphore(1)
        self._save_semaphore = mp.Semaphore(1)
        self._omit_nones = omit_nones

    def _save_history(self, crawler_metric_seq, step_seq):
        pbar = tqdm(total=len(crawler_metric_seq), desc='Saving history')
        for crawler in crawler_metric_seq.keys():
            metric_seq = crawler_metric_seq[crawler]
            for metric in metric_seq.keys():
                metric_value = dict(zip(step_seq, metric_seq[metric]))
                # If all values are None, omit saving
                if self._omit_nones and all(v is None for v in metric_value.values()):
                    continue

                path_pattern = ResultsMerger.names_to_path(
                    self.graph.full_name,
                    declaration_to_filename(crawler.declaration),
                    declaration_to_filename(metric.declaration), self.budget)
                directory = os.path.dirname(path_pattern)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                path = ResultsMerger.next_file(path_pattern.parent)

                with open(path, 'w') as f:
                    # NOTE: will raise error for numpy values
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

        pbar = tqdm(total=self.budget, desc='Running iterations', position=0, leave=True)

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
