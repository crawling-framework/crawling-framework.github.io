import os
import subprocess, sys
import logging

from crawlers.cbasic import filename_to_definition
from graph_io import konect_names, GraphCollections, netrepo_names, other_names
from running.history_runner import CrawlerHistoryRunner
from running.metrics_and_runner import TopCentralityMetric
from statistics import Stat

clouds = [
    'ubuntu@83.149.198.220',
    'ubuntu@83.149.198.231',
]

local_dir = '/home/misha/workspace/crawling'
remote_dir = '/home/ubuntu/workspace/crawling'
ssh_key = '~/.ssh/drobyshevsky_home_key.pem'


def cloud_io():

    # Copy stats to remote

    names = ['ca-citeseer', 'ca-dblp-2010', 'rec-amazon', 'rec-github', 'sc-pkustk13', 'soc-BlogCatalog',
             'soc-brightkite', 'soc-slashdot', 'soc-themarker', 'socfb-Bingham82', 'socfb-OR',
             'socfb-Penn94', 'socfb-wosn-friends', 'tech-RL-caida', 'tech-p2p-gnutella', 'web-arabic-2005']
    cloud = clouds[0]
    collection = 'konect'
    for name in ['slashdot-threads']:
    # for name in ['web-uk-2005', 'web-italycnr-2000', 'ca-dblp-2012', 'sc-pwtk']:
        # if not os.path.exists('%s/data/%s/%s.ij_stats/EccDistr' % (local_dir, collection, name)):
        #     continue

        # loc2rem_copy_command = 'scp -i %s -r %s/data/%s/%s.ij_stats/ %s:%s/data/%s/' % (
        #     ssh_key, local_dir, collection, name, cloud, remote_dir, collection)
        # loc2rem_copy_command = 'scp -i %s -r %s/results/k=0.01/%s/ %s:%s/results/k=0.01/' % (
        #     ssh_key, local_dir, name, cloud, remote_dir, )

        # rem2loc_copy_command = 'scp -i %s -r %s:%s/data/%s/%s.ij_stats/EccDistr %s/data/%s/%s.ij_stats/' % (
        #     ssh_key, cloud, remote_dir, collection, name, local_dir, collection, name)
        rem2loc_copy_command = 'scp -i %s -r %s:%s/results/k=0.01/ %s/%s/' % (
            ssh_key, cloud, remote_dir, local_dir, cloud)

        command = rem2loc_copy_command

        logging.info("Executing command: '%s' ..." % command)
        retcode = subprocess.Popen(command, shell=True, stdout=sys.stdout, stderr=sys.stderr).wait()
        if retcode != 0:
            logging.error("returned code = %s" % retcode)
            raise RuntimeError("unsuccessful: '%s'" % command)
        else:
            logging.info("OK")


    # # Run script
    # cloud = cloud2
    # connect_command = 'ssh %s -i %s' % (cloud2, ssh_key)
    # logging.info("Executing command: '%s' ..." % connect_command)
    # retcode = subprocess.Popen(connect_command, shell=True, stdout=sys.stdout, stderr=sys.stderr).wait()
    #
    # run_command = 'cd workspace/crawling/; PYTHONPATH=/home/ubuntu/workspace/crawling/src python3 src/experiments/crawlers_runs.py'
    # logging.info("Executing command: '%s' ..." % run_command)
    # retcode = subprocess.Popen(run_command, shell=True, stdout=sys.stdout, stderr=sys.stderr).wait()


def do_remote(host: str, command: str, ignore_fails=False):
    """
    Run the command on the remote host.

    :param host: host name e.g. 'ubuntu@83.149.198.220'
    :param command: string e.g. 'ls'
    :return: subprocess.Popen exit code, 0 means success
    """
    ssh_commands = ['ssh', '-i', ssh_key, host]

    # Check connection
    # retcode = subprocess.Popen(ssh_commands, shell=False, stdout=sys.stdout, stderr=sys.stderr).wait()

    print("%s: %s" % (host, command))

    commands = ssh_commands + command.split()
    # logging.debug("Executing command: '%s' ..." % command)
    retcode = subprocess.Popen(commands, shell=False, stdout=sys.stdout, stderr=sys.stderr).wait()
    if retcode != 0:
        err_msg = "exit code %s for command '%s'" % (retcode, command)
        if ignore_fails:
            logging.exception(err_msg)
        else:
            raise RuntimeError(err_msg)
    else:
        logging.debug("OK")
    return retcode


def copy_local2remote(host: str, src: str, dst: str, ignore_fails=False):
    """
    Copy files from local to remote host via scp command.

    :param host: remote host
    :param src: full source path
    :param dst: full destination path
    :param ignore_fails:
    :return:
    """
    commands = ['scp', '-i', ssh_key]
    if os.path.isdir(src):
        commands.append('-r')
    commands += [src, "%s:%s" % (host, dst)]
    command = ' '.join(commands)

    print(command)
    retcode = subprocess.Popen(command, shell=True, stdout=sys.stdout, stderr=sys.stderr).wait()
    if retcode != 0:
        err_msg = "exit code %s for command '%s'" % (retcode, command)
        if ignore_fails:
            logging.exception(err_msg)
        else:
            raise RuntimeError(err_msg)
    else:
        logging.info("OK")
    return retcode


def copy_remote2local(host: str, src: str, dst: str, ignore_fails=False):
    """
    Copy files from remote to local host via scp command.

    :param host: remote host
    :param src: full source path
    :param dst: full destination path
    :param ignore_fails:
    :return:
    """
    commands = ['scp', '-i', ssh_key]
    if os.path.isdir(src):
        commands.append('-r')
    commands += [src, "%s:%s" % (host, dst)]
    command = ' '.join(commands)

    print(command)
    retcode = subprocess.Popen(command, shell=True, stdout=sys.stdout, stderr=sys.stderr).wait()
    if retcode != 0:
        err_msg = "exit code %s for command '%s'" % (retcode, command)
        if ignore_fails:
            logging.exception(err_msg)
        else:
            raise RuntimeError(err_msg)
    else:
        logging.info("OK")
    return retcode


def cloud_prepare(host: str):
    # pass
    # Pull from branch cloud
    do_remote(host, 'eval `ssh-agent -s`; ssh-add ~/.ssh/cloud_rsa; cd workspace/crawling; git checkout cloud; git pull')

    # # Copy graphs and stats
    # do_remote(host, "cd workspace/crawling/data; mkdir konect; mkdir netrepo; mkdir other", ignore_fails=True)
    # for name in konect_names + netrepo_names + other_names:
    #     # Graph stats
    #     src = GraphCollections.get(name, not_load=True).path + '_stats'
    #     dst = os.path.dirname(src.replace('misha', 'ubuntu'))
    #     do_remote(host, "cd workspace/crawling; mkdir '%s'" % dst, ignore_fails=True)
    #     copy_local2remote(host, src=src, dst=dst)
    #     # Graph
    #     src = GraphCollections.get(name, not_load=True).path
    #     dst = os.path.dirname(src.replace('misha', 'ubuntu'))
    #     copy_local2remote(host, src=src, dst=dst)

    # pp = 'PYTHONPATH=~/workspace/crawling/src python3'
    # for name in netrepo_names:
    #     command = ' '.join([pp, '/home/ubuntu/workspace/crawling/src/statistics.py -n', "'%s'" % name, '-s NODES'])
    #     do_remote(host, command)


def cloud_run(host: str):
    pp = 'PYTHONPATH=~/workspace/crawling/src python3'
    command = ' '.join([pp, '/home/ubuntu/workspace/crawling/src/experiments/cloud_run.py'])
    do_remote(host, command)


def main():
    from crawlers.cbasic import RandomCrawler, RandomWalkCrawler, BreadthFirstSearchCrawler, \
        DepthFirstSearchCrawler, SnowBallCrawler, MaximumObservedDegreeCrawler, PreferentialObservedDegreeCrawler
    from crawlers.cadvanced import DE_Crawler
    from crawlers.multiseed import MultiInstanceCrawler

    batches = [1, 10, 100, 1000, 10000]
    multi_counts = [2, 3, 5, 10, 30, 100, 1000]
    crawler_defs = [
        (RandomWalkCrawler, {}),
        (RandomCrawler, {}),
        (BreadthFirstSearchCrawler, {}),
        (DepthFirstSearchCrawler, {}),
        (SnowBallCrawler, {'p': 0.1}),
        (SnowBallCrawler, {'p': 0.25}),
        (SnowBallCrawler, {'p': 0.75}),
        (SnowBallCrawler, {'p': 0.9}),
        (SnowBallCrawler, {'p': 0.5}),
    ] + [
        (MaximumObservedDegreeCrawler, {'batch': b}) for b in batches
    ] + [
        (PreferentialObservedDegreeCrawler, {'batch': b}) for b in batches
    ] + [
        (DE_Crawler, {}),
    ] + [
        (MultiInstanceCrawler, {'count': c, 'crawler_def': (BreadthFirstSearchCrawler, {})}) for c in multi_counts
    ] + [
        (MultiInstanceCrawler, {'count': c, 'crawler_def': (MaximumObservedDegreeCrawler, {'batch': 1})}) for c in multi_counts
    ] + [
        (MultiInstanceCrawler, {'count': c, 'crawler_def': (PreferentialObservedDegreeCrawler, {'batch': 1})}) for c in multi_counts
    ] + [
        (MultiInstanceCrawler, {'count': c, 'crawler_def': (DE_Crawler, {})}) for c in multi_counts
    ]

    p = 0.001
    metric_defs = [
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.DEGREE_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.PAGERANK_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.BETWEENNESS_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.ECCENTRICITY_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.CLOSENESS_DISTR.short}),
        (TopCentralityMetric, {'top': p, 'measure': 'Re', 'part': 'crawled', 'centrality': Stat.K_CORENESS_DISTR.short}),
    ]

    n_instances = 8
    for graph_name in netrepo_names:
        g = GraphCollections.get(graph_name)
        chr = CrawlerHistoryRunner(g, crawler_defs, metric_defs)
        chr.run_missing(n_instances, max_cpus=8, max_memory=30)
        print('\n\n')


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    # cloud_io()
    # cloud_prepare(clouds[0])
    # cloud_run(clouds[0])

    main()  # to be run from cloud