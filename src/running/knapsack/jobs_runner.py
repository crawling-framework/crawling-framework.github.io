import datetime
import json

import matplotlib.pyplot as plt
import numpy as np
import logging
import multiprocessing as mp
import traceback
from random import randint
from time import time, sleep

from tqdm import tqdm

from running.knapsack.jobs_balancer import LoadBalancer, LoadState, Job, RandomBalancer
from running.knapsack.simulations import LoadSimulator


def ram_usage(pid):
    """ RAM usage in MBytes """
    with open(f'/proc/{pid}/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]
    return int(memusage.strip()) / 1024


class JobProcess(mp.Process, Job):
    """ multiprocessing.Process that handles exceptions.
    From https://stackoverflow.com/a/33599967/8900030
    """

    def __init__(self, cpu=None, ram=None, time=None, id=None, at_end=None,
                 group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        if id is None:
            id = '(' + ','.join(str(x) for x in args) + ','.join(f"{k}={v}" for k, v in kwargs.items()) + ')'
        Job.__init__(self, cpu, ram, time, id)
        mp.Process.__init__(
            self, group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)
        self._kwargs = kwargs
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None
        self.at_end = at_end

    def run(self):
        try:
            mp.Process.run(self)
            # Report RAM usage
            self.report_ram()
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # send_vk(self._pconn.recv()[1])
            raise e

    def report_ram(self):
        """ Report RAM usage """
        stat = {
            'id': self.id,
            'ram': ram_usage(self.ident),
            'measured at': str(datetime.datetime.now()),
        }
        with open('ram_usage', 'a') as f:
            f.write(json.dumps(stat, indent=1))
            f.write('\n')

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

    def end(self):
        """ Call function at the end of job """
        if self.at_end is not None:
            self.at_end()


class JobsRunner(LoadSimulator):
    """ Runs a set of jobs
    """
    def __init__(self, cpus, ram, balancer_class):
        super().__init__(cpus, ram)
        self.balancer_class = balancer_class

        # self.finished_jobs = set()  # Set of already finished jobs
        self.running_jobs = set()  # Set of currently running jobs
        self.start_time = 0

    def work(self):
        """ Execute jobs until one is ended. """
        if not self._running_jobs:
            raise RuntimeError("No jobs to execute")

        while True:
            for job in self.running_jobs:
                assert isinstance(job, JobProcess)
                if not job.is_alive():
                    self.current_time = time() - self.start_time
                    self.job_end(job)
                    print("Finished job", job)
                    return self.current_time, job

            # All are in work
            sleep(0.1)

    def put(self, job):
        super(JobsRunner, self).put(job)
        self.running_jobs.add(job)
        job.start()

    def job_end(self, job):
        super(JobsRunner, self).job_end(job)
        self.running_jobs.remove(job)
        # self.finished_jobs.add(job)
        job.end()

    def run(self, jobs):
        for job in jobs:
            assert isinstance(job, JobProcess)

        balancer = self.balancer_class(self.cpus, self.ram, jobs)
        balancer.balance()

        self.start_time = time()
        for job in tqdm(balancer, desc="Executing jobs"):
            # Wait until available resources appear
            if job is None:
                if not self.has_jobs:
                    raise RuntimeError("Nothing is running but a next Job can't be added!")

                self.current_time, finished_job = self.work()
                balancer.job_is_done(finished_job, self.current_time)
            else:
                # Sample job execution time
                logging.info('Start job %s' % job)
                self.put(job)

        while self.has_jobs:
            self.current_time, finished_job = self.work()
            balancer.job_is_done(finished_job, self.current_time)

        print("Total time", balancer.__class__.__name__, self.current_time)
