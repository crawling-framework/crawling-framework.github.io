import copy
from heapq import heappush, heappop, heapify
from abc import abstractmethod, ABCMeta
from collections import deque
from itertools import permutations
from queue import PriorityQueue
from time import time
import numpy as np


class Job:
    """ A job to do.
    Contains prior estimations of its parameters.

    * cpu -- max number of CPUs is uses
    * ram -- max amount of RAM it uses, in Mb
    * time -- max time it takes to be computed, in seconds

    Any parameter could be None, meaning it is unknown.
    """
    def __init__(self, cpu=None, ram=None, time=None, id=None):
        self.id = id
        self.cpu = cpu
        self.ram = ram
        self.time = time

    def __str__(self):
        if self.time is None:
            time_str = "?"
        elif self.time >= 1e6:
            time_str = "%.0f" % self.time
        else:
            time_str = ("%f" % self.time)[:6]
        return f"Job{self.id} CPUs={self.cpu}, RAM={self.ram}, time={time_str}"

    def __lt__(self, other):
        """ To compare in priority queue """
        return True  # order is undefined

    def run(self, *args, **kwargs):
        raise NotImplementedError()


class LoadState:
    """ Imitation of CPU and RAM loading with jobs.
    Remembers for each Job which CPUs it used, amount of RAM, and its start and end times.
    """

    def __init__(self, cpus, ram):
        self.cpus = cpus
        self.ram = ram

        self.current_time = 0
        self._running_jobs = 0
        self.free_cpus = self.cpus
        self.free_ram = self.ram
        self._end_times = []  # (end time, job)

    def reset(self):
        """ Clear resources, set timer to zero. """
        self.current_time = 0
        self._end_times = []
        self.free_cpus = self.cpus
        self.free_ram = self.ram

    def clone(self):
        state = LoadState(self.cpus, self.ram)
        state.free_cpus = self.free_cpus
        state.free_ram = self.free_ram
        state.current_time = self.current_time
        state._running_jobs = self._running_jobs
        state._end_times = list(self._end_times)
        return state

    @property
    def has_jobs(self):
        return self._running_jobs > 0

    def can_put(self, job):
        """ Check whether possible to add the specified job """
        return self.free_ram >= job.ram and self.free_cpus >= job.cpu

    def put(self, job):
        """ Start running a next job """
        self.free_cpus -= job.cpu
        self.free_ram -= job.ram
        assert self.free_ram >= 0 and self.free_cpus >= 0
        job_time = np.infty if job.time is None else job.time
        heappush(self._end_times, (self.current_time + job_time, job))
        self._running_jobs += 1

    def work(self):
        """ Move to the moment the next job is ended. """
        if not self._running_jobs:
            raise StopIteration
        self.current_time, finished_job = heappop(self._end_times)
        self.job_end(finished_job)
        return self.current_time, finished_job

    def job_end(self, job):
        """ Free resources from a finished job"""
        self.free_cpus += job.cpu
        self.free_ram += job.ram
        self._running_jobs -= 1


class LoadBalancer:
    """ Parent class for jobs balancing algorithms
    """
    __metaclass__ = ABCMeta

    def __init__(self, cpus, ram, jobs: list):
        """
        Args:
            cpus: max available CPUs
            ram: available max RAM, in Mb
            jobs: list of jobs to balance
        """
        self.cpus = cpus
        self.ram = ram
        self.load_state = LoadState(cpus, ram)
        self.jobs = jobs

        self.finished_jobs = set()  # Set of already finished jobs
        self.running_jobs = set()  # Set of currently running jobs

    @abstractmethod
    def balance(self):
        """ Balance a set of jobs and yields in some order.
        Called at the beginning.
        """
        raise NotImplementedError()

    # def has_next(self):
    #     return len(self.waiting_jobs) > 0

    def __next__(self):
        """ Get the next job to do or None if it's early yet.
        """
        raise NotImplementedError()

    def __iter__(self):
        """ Iterate all jobs in a specific order
        """
        while len(self.finished_jobs) + len(self.running_jobs) < len(self.jobs):
            yield next(self)

    def job_is_done(self, job, current_time=None):
        """ When called, JobsBalancer knows that specified job is finished.
        current_time can be specified in a simulation study.
        """
        self.load_state.current_time = time() if current_time is None else current_time
        self.load_state.job_end(job)
        self.finished_jobs.add(job)
        self.running_jobs.remove(job)


class MultiBalancer(LoadBalancer):
    """ Does the balancing after each job is finished.
    Suits when jobs parameters are not fully known a priory.
    balance() will be called again each time a job ends.
    """
    def __init__(self, cpus, ram, jobs: list):
        super().__init__(cpus, ram, jobs)

    def job_is_done(self, job, current_time=None):
        super(MultiBalancer, self).job_is_done(job, current_time=current_time)
        # Do a re-balancing
        self.balance()


class RandomBalancer(LoadBalancer):
    """ Order of jobs is random
    """
    def __init__(self, cpus, ram, jobs: list, shuffle=False):
        super().__init__(cpus, ram, jobs)
        self.waiting_jobs = deque()  # A queue of jobs in the order for execution
        self.shuffle = shuffle

    def balance(self):
        if self.shuffle:
            np.random.shuffle(self.jobs)
        self.waiting_jobs.extend(self.jobs)

    def __next__(self):
        """ Get the next job to do or None if it's early yet.
        """
        if len(self.waiting_jobs) == 0:
            raise StopIteration
        job = self.waiting_jobs[0]
        if not self.load_state.can_put(job):
            return None
        job = self.waiting_jobs.popleft()
        self.load_state.put(job)
        self.running_jobs.add(job)
        return job


class GreedyBalancer(LoadBalancer):
    """ Most weighted job first
    """
    def __init__(self, cpus, ram, jobs):
        super().__init__(cpus, ram, jobs)
        self.weighted_jobs = []

    def balance(self):
        """ Greedy order. """
        # FIXME List is slow for removal
        max_time = max(j.time if j.time is not None else 1 for j in self.jobs)
        self.weighted_jobs = sorted(
            [(job.cpu * job.ram * (job.time if job.time is not None else max_time), job)
             for job in self.jobs], reverse=True)

    def __next__(self):
        if len(self.weighted_jobs) == 0:
            raise StopIteration

        for i, (w, job) in enumerate(self.weighted_jobs):
            if self.load_state.can_put(job):
                del self.weighted_jobs[i]
                self.load_state.put(job)
                self.running_jobs.add(job)
                return job
        return None


class FullSearchBalancer(LoadBalancer):
    """ Optimal load based on an exhaustive search
    """
    def __init__(self, cpus, ram, jobs):
        super().__init__(cpus, ram, jobs)
        self.waiting_jobs = []

    def balance(self):
        """ Greedy order. """
        if len(self.jobs) > 10:
            # Too long
            raise NotImplementedError()

        # DFS
        def step(running_jobs: list, left_jobs, load_state: LoadState, least_time):
            if len(left_jobs) == 0:
                while load_state.has_jobs:
                    load_state.work()
                # print([str(j) for j in running_jobs], load_state.current_time)
                if load_state.current_time < least_time[0]:
                    self.waiting_jobs = running_jobs
                    least_time[0] = load_state.current_time
            else:
                for j in left_jobs:
                    new_running_jobs = running_jobs + [j]
                    new_left_jobs = set(left_jobs)
                    new_left_jobs.remove(j)
                    new_load_state = load_state.clone()
                    while not new_load_state.can_put(j):
                        new_load_state.work()
                        if new_load_state.current_time >= least_time[0]:
                            continue
                    new_load_state.put(j)
                    step(new_running_jobs, new_left_jobs, new_load_state, least_time)

        least_time = [1e99]
        # step([], self.jobs, self.load_state, least_time)

        # Linear
        for jobs_order in permutations(self.jobs, len(self.jobs)):
            load_state = LoadState(self.cpus, self.ram)
            for j in jobs_order:
                while not load_state.can_put(j):
                    load_state.work()
                    # if load_state.current_time >= least_time[0]:
                    #     continue
                load_state.put(j)
            while load_state.has_jobs:
                load_state.work()
            if load_state.current_time < least_time[0]:
                self.waiting_jobs = jobs_order
                least_time[0] = load_state.current_time

        self.waiting_jobs = list(self.waiting_jobs)

    def __next__(self):
        if len(self.waiting_jobs) == 0:
            raise StopIteration
        job = self.waiting_jobs[0]
        if not self.load_state.can_put(job):
            return None
        del self.waiting_jobs[0]  # FIXME list is not optimal
        self.load_state.put(job)
        self.running_jobs.add(job)
        return job

