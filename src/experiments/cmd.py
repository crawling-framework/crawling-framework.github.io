from argparse import ArgumentParser

from experiments.paper_experiments import paper_configs
from running.history_runner import SmartCrawlersRunner
from running.metrics import OracleBasedMetric
from search.feature_extractors import NeighborsFeatureExtractor
from search.oracles import HasAttrValueOracle
from search.predictor_based_crawlers.mab import MultiPredictor, \
    ExponentialDynamicWeightsMultiPredictorCrawler, BetaDistributionMultiPredictorCrawler, \
    FollowLeaderMABCrawler
from search.predictor_based_crawlers.predictor_based import PredictorBasedCrawler
from search.predictor_based_crawlers.training_strategies import BoostingTrainStrategy
from search.predictors.simple_predictors import MaximumTargetNeighborsPredictor, SklearnPredictor


def run(g, c, n):
    g_config = {pc[4]: pc for pc in paper_configs}
    config = g_config[g]
    graph_names = [config[0]]
    attr = config[1]
    value = config[2]
    budget = config[3]
    title = config[4]
    oracle = HasAttrValueOracle(attribute=attr, value=value)

    # Define predictors
    from search.predictors.gnn_predictors import GNNPredictor
    nfe = NeighborsFeatureExtractor(od=True, cc=True, cnf=True, tnf=True, tri=True, neighs1=True, hist=5)
    mtn = MaximumTargetNeighborsPredictor()
    xgb = SklearnPredictor("GradientBoostingClassifier", nfe, name="XGB")
    knn = SklearnPredictor("KNeighborsClassifier", nfe, algorithm='kd_tree', n_neighbors=30, name="KNN")
    rf = SklearnPredictor("RandomForestClassifier", nfe, n_estimators=100, name="RF")
    svc = SklearnPredictor("SVC", nfe, kernel='rbf', probability=True, name="SVC")
    sage = GNNPredictor('SAGEConv', (5,), aggregator_type='gcn', attributes=[], epochs=200, name="SAGE")
    gat1 = GNNPredictor('GATConv', (5,), num_heads=1, merge='mean', attributes=[], epochs=200, name="GAT-1")
    gat3 = GNNPredictor('GATConv', (5,), num_heads=3, merge='mean', attributes=[], epochs=200, name="GAT-3")
    multipred5 = MultiPredictor([mtn, xgb, knn, rf, svc], name="Multi-5")
    multipred6 = MultiPredictor([mtn, xgb, knn, rf, svc, sage], name="Multi-6")

    predictor = None
    crawler = None
    trainer = None if c == 'MTN' else BoostingTrainStrategy(name="boost")
    if c in ['MTN', 'XGB', 'RF', 'KNN', 'SVC', 'SAGE', 'GAT-1', 'GAT-3']:
        predictor = {
            'MTN': mtn,
            'XGB': xgb,
            'KNN': knn,
            'RF': rf,
            'SVC': svc,
            'SAGE': sage,
            'GAT-1': gat1,
            'GAT-3': gat3,
        }[c]
        crawler = PredictorBasedCrawler
    elif c in ['DW-5', 'FL-5', 'TS-5', 'DW-6', 'FL-6']:
        if c[-1] == '5':
            predictor = multipred5
        elif c[-1] == '6':
            predictor = multipred6
        crawler = {
            'DW': ExponentialDynamicWeightsMultiPredictorCrawler,
            'FL': FollowLeaderMABCrawler,
            'TS': BetaDistributionMultiPredictorCrawler,
        }[c[:2]]
    assert predictor
    assert crawler

    # Declarations
    crawler_decls = [
        (crawler, {
            'predictor': predictor.declaration,
            'training_strategy': trainer,
            're_estimate': 'after_train', 'initial_seed': 'target', 'oracle': oracle,
            'name': c,
        })
    ]

    metric_decls = [
        (OracleBasedMetric, {'oracle': oracle, 'measure': 'size', 'part': 'crawled'}),
    ]

    mcr = SmartCrawlersRunner(graph_names, crawler_decls, metric_decls, budget)
    mcr.run_adaptive(n)

    # # Show the results
    from running.merger import ResultsMerger
    crm = ResultsMerger(graph_names, crawler_decls, metric_decls, budget, x_lims=(0, budget))
    crm.draw_by_crawler(x_normalize=False, draw_each_instance=True, draw_error=True, scale=8)
    crm.draw_aggregated(aggregator='TC', scale=3, xticks_rotation=90)
    crm.show_plots()


if __name__ == '__main__':
    graphs = list(pc[4] for pc in paper_configs)
    crawlers = ['MTN', 'XGB', 'RF', 'KNN', 'SVC', 'SAGE', 'DW-5', 'FL-5', 'TS-5', 'DW-6', 'FL-6']
    parser = ArgumentParser(description='Social Graph Crawler')
    parser.add_argument('-g', default='donors', help='Graph name', choices=graphs)
    parser.add_argument('-c', default='MTN', help='Crawler name', choices=crawlers)
    parser.add_argument('-n', default=4, help='Number of runs')
    args = parser.parse_args()

    assert args.g in graphs
    assert args.c in crawlers
    assert args.n.isnumeric()

    run(g=args.g, c=args.c, n=int(args.n))
