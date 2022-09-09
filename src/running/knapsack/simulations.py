from random import randint
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from tqdm import tqdm

from running.knapsack.jobs_balancer import Job, RandomBalancer, LoadBalancer, LoadState, \
    GreedyBalancer, FullSearchBalancer


class LoadHistory():
    """ Job -> [start time, end time, loaded cpus]
    """
    def __init__(self):
        self._dict = {}

    def clear(self):
        self._dict.clear()

    def job_start(self, job, loaded_cpus, time):
        self._dict[job] = [time, None, loaded_cpus]
        # print(f"Job {job} starts at {time}")

    def job_end(self, job, time):
        self._dict[job][1] = time  # Set end time
        # print(f"Job {job} ends at {time}")

    def loaded_cpus(self, job):
        return self._dict[job][2]

    def __iter__(self):
        for job, info in self._dict.items():
            yield job, info

    def __str__(self):
        res = []
        for job, (start, end, loaded_cpus) in self._dict.items():
            res.append(f"{'%.1f' % start} -- {'%.1f' % end} CPUs={loaded_cpus} - '{job}'")
        return '\n'.join(res)


class LoadSimulator(LoadState):
    def __init__(self, cpus, ram):
        super().__init__(cpus, ram)

        self.cpu_busy = [False] * self.cpus  # cpu -> is loaded
        self.history = LoadHistory()

    def reset(self):
        super(LoadSimulator, self).reset()
        self.history.clear()
        for c in range(self.cpus):
            self.cpu_busy[c] = False

    def put(self, job):
        super(LoadSimulator, self).put(job)
        loaded_cpus = []
        for c in range(self.cpus):
            if not self.cpu_busy[c]:
                # Load next cpu
                self.cpu_busy[c] = True
                loaded_cpus.append(c)
                if len(loaded_cpus) == job.cpu:
                    break
        self.history.job_start(job, loaded_cpus, self.current_time)

    def job_end(self, job):
        super(LoadSimulator, self).job_end(job)
        for c in self.history.loaded_cpus(job):
            self.cpu_busy[c] = False
        self.history.job_end(job, self.current_time)

    def simulate(self, balancer: LoadBalancer):
        self.reset()

        for job in tqdm(balancer, desc="Simulating jobs running"):
            # Wait until available resources appear
            if job is None:
                if not self.has_jobs:
                    raise RuntimeError("Nothing is running but a next Job can't be added!")

                self.current_time, finished_job = self.work()
                balancer.job_is_done(finished_job, self.current_time)
            else:
                # Sample job execution time
                if job.time is None:
                    job.time = np.random.gamma(1, 3600)
                self.put(job)

        while self.has_jobs:
            self.current_time, finished_job = self.work()
            balancer.job_is_done(finished_job, self.current_time)

        print("Total time", balancer.__class__.__name__, self.current_time)

    def draw(self):
        COLORS = ['black', 'b', 'g', 'r', 'c', 'm', 'y',
                  'darkblue', 'darkgreen', 'darkred', 'darkmagenta', 'darkorange', 'darkcyan',
                  'pink', 'lime', 'wheat', 'lightsteelblue']

        plt.style.use('seaborn')
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # Plot RAMs
        ram_points = []  # (time, ram delta)
        for job, (start, end, loaded_cpus) in self.history:
            ram_points.append((start, job.ram))
            ram_points.append((end, -job.ram))
        ram_points.sort()
        total = 0
        xs = []
        ys = []
        for time, delta in ram_points:
            xs.append(time)
            total += delta
            ys.append(total)

        axes[0].set_ylabel("RAM, Mbytes")
        axes[0].fill_between(xs, ys, step='post')

        axes[0].plot([0, self.current_time], [self.ram, self.ram], color='r',
                     label="max RAM")
        axes[0].legend(frameon=True, loc=1)

        # Plot CPUs
        cpu_bars = []  # list of TL, BL, BR, TR
        colors = []
        job_color = {}
        for job, (start, end, loaded_cpus) in self.history:
            # x = mdates.date2num(start)
            x = start
            w = end - start
            if job not in job_color:
                job_color[job] = COLORS[len(job_color) % len(COLORS)]
            for cpu in loaded_cpus:
                b = cpu - 0.4
                t = cpu + 0.4
                cpu_bars.append(((x, b), (x, t), (x + w, t), (x + w, b)))
                colors.append(job_color[job])

        useAA = True,  # use tuple here
        lw = 0.5,  # and here
        barCollection = PolyCollection(
            cpu_bars, facecolors=colors, edgecolors=colors, antialiaseds=useAA, linewidths=lw)

        axes[1].add_collection(barCollection)
        axes[1].set_yticks(np.arange(self.cpus))
        axes[1].set_ylabel("CPU")
        axes[1].set_xlabel("Time spent")
        axes[1].set_ylim((-0.5, self.cpus - 0.5))

        axes[1].legend(handles=[
            axes[1].plot([], [], marker="s", ms=10, ls="", color=color, label=type)[0]
            for type, color in job_color.items()
        ], frameon=True)
        # locator = mdates.AutoDateLocator(minticks=20, maxticks=40)
        # formatter = mdates.ConciseDateFormatter(locator)
        # axes[0, 0].xaxis.set_major_locator(locator)
        # axes[0, 0].xaxis.set_major_formatter(formatter)
        plt.tight_layout()
