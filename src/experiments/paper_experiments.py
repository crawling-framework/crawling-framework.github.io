from running.history_runner import SmartCrawlersRunner
from running.metrics import OracleBasedMetric
from search.feature_extractors import NeighborsFeatureExtractor
from search.oracles import HasAttrValueOracle
from search.predictor_based_crawlers.mab import MultiPredictor, \
    ExponentialDynamicWeightsMultiPredictorCrawler, BetaDistributionMultiPredictorCrawler, \
    FollowLeaderMABCrawler
from search.predictor_based_crawlers.predictor_based import PredictorBasedCrawler
from search.predictor_based_crawlers.training_strategies import OnlineTrainStrategy, \
    BoostingTrainStrategy
from search.predictors.simple_predictors import MaximumTargetNeighborsPredictor, SklearnPredictor


vk_samples = [
    'vk2-ff20-N10000-A.1611943634',  # N=9998, E=237 057,   avg_deg=47.4,  Q=0.687,   0
    'vk2-ff40-N10000-A.1612175225',  # N=9996, E=709 388,   avg_deg=141.8,  Q=0.622,  1
    'vk2-ff60-N10000-A.1612176028',  # N=9999, E=1 481 512, avg_deg=296.2, Q=0.396,   2
    'vk2-ff80-N10000-A.1612176849',  # N=9991, E=73 037,    avg_deg=14.6,   Q=0.392,  3

    'vk2-ff20-N30000-A.1612280934',  # N=29 997, E=663 130,   avg_deg=44.2,  Q=0.725, 4
    'vk2-ff40-N30000-A.1612271392',  # N=29 996, E=2 022 919, avg_deg=134.8, Q=0.697, 5
    'vk2-ff60-N30000-A.1612271398',  # N=30 000, E=2 173 647, avg_deg=144.8, Q=0.650, 6
    'vk2-ff80-N30000-A.1612263097',  # N=30 000, E=2 140 223, avg_deg=142.6, Q=0.467, 7
    'vk2-ff80-N30000-A.1612271409',  # N=29 999, E=2 297 705, avg_deg=153.0, Q=0.670, 8

    'vk2-ff20-N100000-A.1612177946',  # N=99 998,  E=2 740 416,  avg_deg=54.8,  Q=0.721,  9
    'vk2-ff40-N100000-A.1612175945',  # N=99 996,  E=4 022 715,  avg_deg=80.4,  Q=0.688,  10
    'vk2-ff60-N100000-A.1612175951',  # N=100 000, E=6 337 408,  avg_deg=126.6,  Q=0.677, 11
    'vk2-ff80-N100000-A.1612175956',  # N=99 999,  E=10 752 258, avg_deg=215.0, Q=0.760,  12

    'vk2-ff20-N300000-A.1612185906',  # N=300 000, E=8 255 107,  avg_deg=55.0,  Q=0.773,  13
    'vk2-ff40-N300000-A.1612186139',  # N=300 000, E=5 085 681,  avg_deg=33.8,  Q=0.701,  14
    'vk2-ff60-N300000-A.1612186146',  # N=300 000, E=12 942 557, avg_deg=86.2,  Q=0.817,  15
    'vk2-ff80-N300000-A.1612185660',  # N=299 997, E=26 028 289, avg_deg=173.4,  Q=0.822, 16
]

paper_configs = [  # graph, attribute, value, budget
    [('sel_harv', 'donors'), ('comm_56',), 1, 100, "donors"],  # 0
    [('snap', 'livejournal'), ('comm_1441',), 1, 1200, "livejournal"],  # 1
    [('snap', 'dblp'), ('comm_7556',), 1, 1200, "dblp"],  # 2 - Test
    [('sel_harv', 'kickstarter'), ('comm_1455',), 1, 700, "kickstarter"],  # 3 - Test

    [('attributed', 'vk_10_classes'), ('Программирование',), 1, 1000, "VK-prog"],  # 4
    [('attributed', 'vk_10_classes'), ('Бизнес',), 1, 1000, "VK-busi"],  # 5
    [('attributed', 'vk_10_classes'), ('Здоровье',), 1, 3000, "VK-heal"],  # 6

    [('attributed', 'vk_10_classes'), ('Вязание',), 1, 3000, "VK-knit"],    # 7 - Test
    [('attributed', 'vk_10_classes'), ('Музыка',), 1, 3000, "VK-music"],    # 8 - Test
    [('attributed', 'vk_10_classes'), ('Политика',), 1, 3000, "VK-polit"],  # 9 - Test
    [('attributed', 'vk_10_classes'), ('Спорт',), 1, 3000, "VK-sport"],     # 10 - Test
    [('attributed', 'vk_10_classes'), ('Философия',), 1, 3000, "VK-phil"],  # 11 - Test
    [('attributed', 'vk_10_classes'), ('Цитаты',), 1, 3000, "VK-cita"],     # 12 - Test
    [('attributed', 'vk_10_classes'), ('Юмор',), 1, 3000, "VK-humor"],      # 13 - Test

    [('vk_samples', vk_samples[0]), ('sex',), 1, 1000, "VK-sex"],  # 14
    [('vk_samples', vk_samples[10]), ('sex',), 1, 3000, "VK-sex1"],                 # 15 - Test
    [('vk_samples', vk_samples[10]), ('personal', 'smoking'), 1, 3000, "VK-smok1"],  # 16 - Test
    [('vk_samples', vk_samples[10]), ('personal', 'smoking'), 3, 3000, "VK-smok3"],  # 17 - Test
    [('vk_samples', vk_samples[10]), ('sex',), 2, 3000, "VK-sex2"],  # 18 - Test
    [('vk_samples', vk_samples[10]), ('relation',), 1, 3000, "VK-relat1"],  # 19 - Test
    [('attributed', 'twitter'), ('occupation',), '2', 3000, "TW-occup2"],  # 20 - Test
    [('attributed', 'twitter'), ('occupation',), '7', 3000, "TW-occup7"],  # 21 - Test
]


def single_predictor(config_id=0):
    config = paper_configs[config_id]  # choose graph
    graph_names = [config[0]]
    attr = config[1]
    value = config[2]
    budget = config[3]
    title = config[4]
    oracle = HasAttrValueOracle(attribute=attr, value=value)

    gnn_attributes = []
    # gnn_attributes = [('feature',)]

    # Define predictors
    mtn = MaximumTargetNeighborsPredictor()
    nfe = NeighborsFeatureExtractor(
        od=True, cc=True, cnf=True, tnf=True, tri=True, neighs1=True, hist=5)
    xgb = SklearnPredictor("GradientBoostingClassifier", nfe, name="XGB")
    knn = SklearnPredictor("KNeighborsClassifier", nfe, algorithm='kd_tree', n_neighbors=30, name="KNN")
    rf = SklearnPredictor("RandomForestClassifier", nfe, n_estimators=100, name="RF")
    svc = SklearnPredictor("SVC", nfe, kernel='rbf', probability=True, name="SVC")

    from search.predictors.gnn_predictors import GNNPredictor
    hidden = 5
    epochs = 200
    gcn = GNNPredictor('GraphConv', (hidden,), attributes=gnn_attributes, epochs=epochs, name='GCN')
    sage_mean = GNNPredictor('SAGEConv', (hidden,), aggregator_type='mean',
                             attributes=gnn_attributes, epochs=epochs, name="SAGE")
    sage_gcn = GNNPredictor('SAGEConv', (hidden,), aggregator_type='gcn',
                            attributes=gnn_attributes, epochs=epochs, name="SAGE")
    gat1 = GNNPredictor('GATConv', (hidden,), num_heads=1, merge='mean',
                        attributes=gnn_attributes, epochs=epochs, name="GAT-1")
    gat3 = GNNPredictor('GATConv', (hidden,), num_heads=3, merge='mean',
                        attributes=gnn_attributes, epochs=epochs, name="GAT-3")

    # NOTE: we need .declaration in order to reset the predictor in consequent runs

    # Trainers
    online = OnlineTrainStrategy(name="online")
    online_each = OnlineTrainStrategy(retrain_step_exponent=1.0001, name="online-each")
    boost = BoostingTrainStrategy(name="boost")
    boost1000 = BoostingTrainStrategy(max_boost_iterations=100, train_max_samples=1000, name="boost1000")
    boost3000 = BoostingTrainStrategy(max_boost_iterations=100, train_max_samples=1000, name="boost1000")

    # NOTE: every parameter except Oracle must be a declaration not an object! (to avoid sharing)
    crawler_decls = []
    crawler_decls.extend([
        # (RandomCrawler, {'oracle': oracle, 'initial_seed': 'target', 'name': 'RC-T'}),
        (PredictorBasedCrawler, {
            'predictor': mtn, 'oracle': oracle, 'initial_seed': 'target', 'name': "MOD-T",
        }),
    ])
    crawler_decls.extend([
        (PredictorBasedCrawler, {
            'predictor': predictor.declaration,
            'training_strategy': trainer.declaration,
            're_estimate': 'after_train', 'initial_seed': 'target', 'oracle': oracle,
            'statistics_flags': [
                "crawled_nodes", "target_flags", "seed_estimation", "observed_set_size"],
            'name': predictor.name + "-" + trainer.name,
        })
        for pred in [xgb, rf, knn, svc, gcn, sage_gcn, gat1, gat3]
        for predictor, trainer in [
            # (pred, online),
            (pred, boost),
            ]
    ])

    metric_decls = [
        (OracleBasedMetric, {'oracle': oracle, 'measure': 'size', 'part': 'crawled'}),
    ]

    n_instances = 20

    mcr = SmartCrawlersRunner(graph_names, crawler_decls, metric_decls, budget)
    mcr.run_adaptive(n_instances, max_cpus=2, max_memory=20)

    # # Show the results
    from running.merger import ResultsMerger
    crm = ResultsMerger(graph_names, crawler_decls, metric_decls, budget, x_lims=(0, budget))
    crm.draw_by_crawler(x_normalize=False, draw_each_instance=True, draw_error=True, scale=8)
    crm.draw_aggregated(aggregator='TC', scale=3, xticks_rotation=90)
    crm.show_plots()


def multi_predictor(config_id=0):
    config = paper_configs[config_id]  # choose graph
    graph_names = [config[0]]
    attr = config[1]
    value = config[2]
    budget = config[3]
    title = config[4]
    oracle = HasAttrValueOracle(attribute=attr, value=value)

    gnn_attributes = []
    # gnn_attributes = [('feature',)]

    # Define predictors
    mtn = MaximumTargetNeighborsPredictor()
    nfe = NeighborsFeatureExtractor(
        od=True, cc=True, cnf=True, tnf=True, tri=True, neighs1=True, hist=5)
    xgb = SklearnPredictor("GradientBoostingClassifier", nfe, name="XGB")
    knn = SklearnPredictor("KNeighborsClassifier", nfe, algorithm='kd_tree', n_neighbors=30, name="KNN")
    rf = SklearnPredictor("RandomForestClassifier", nfe, n_estimators=100, name="RF")
    svc = SklearnPredictor("SVC", nfe, kernel='rbf', probability=True, name="SVC")

    from search.predictors.gnn_predictors import GNNPredictor
    gcn = GNNPredictor('GraphConv', (5,), attributes=gnn_attributes, epochs=100, name='GCN')
    sage = GNNPredictor('SAGEConv', (5,), aggregator_type='mean',
                        attributes=gnn_attributes, epochs=200, name="SAGE")
    sage_gcn = GNNPredictor('SAGEConv', (5,), aggregator_type='gcn',
                            attributes=gnn_attributes, epochs=200, name="SAGE")
    gat1 = GNNPredictor('GATConv', (5,), num_heads=1, merge='mean',
                        attributes=gnn_attributes, epochs=200, name="GAT-1")
    gat3 = GNNPredictor('GATConv', (5,), num_heads=3, merge='mean',
                        attributes=gnn_attributes, epochs=200, name="GAT-3")

    # NOTE: we need .declaration in order to reset the predictor in consequent runs
    multipred5 = MultiPredictor([mtn, xgb, knn, rf, svc], name="Multi-5")
    multipred6 = MultiPredictor([mtn, xgb, knn, rf, svc, sage_gcn], name="Multi-6")

    # Trainers
    online = OnlineTrainStrategy(name="online")
    online_each = OnlineTrainStrategy(retrain_step_exponent=1.0001, name="online-each")
    boost = BoostingTrainStrategy(name="boost")
    boost1000 = BoostingTrainStrategy(max_boost_iterations=100, train_max_samples=1000, name="boost1000")
    boost3000 = BoostingTrainStrategy(max_boost_iterations=100, train_max_samples=1000, name="boost1000")

    # NOTE: every parameter except Oracle must be a declaration not an object! (to avoid sharing)
    crawler_decls = []
    crawler_decls.extend([
        (crawler, {
            'predictor': predictor.declaration,
            'training_strategy': trainer.declaration,
            're_estimate': 'after_train', 'initial_seed': 'target', 'oracle': oracle,
            'statistics_flags': [
                "crawled_nodes", "target_flags", "seed_estimation", "observed_set_size"],
            'name': crawler.short + "-" + predictor.name + "-" + trainer.name,
        })
        for crawler in [
            # PredictorBasedCrawler,
            ExponentialDynamicWeightsMultiPredictorCrawler,
            FollowLeaderMABCrawler,
        ]
        # for pred in [multipred5]
        for pred in [multipred6]
        for predictor, trainer in [
            # (pred, online),
            (pred, boost),
            ]
    ])
    crawler_decls.extend([
        (crawler, {
            'predictor': predictor.declaration,
            'training_strategy': trainer.declaration,
            're_estimate': 'after_train', 'initial_seed': 'target', 'oracle': oracle,
            'statistics_flags': [
                "crawled_nodes", "target_flags", "seed_estimation", "observed_set_size"],
            'name': crawler.short + "-" + predictor.name + "-" + trainer.name,
        })
        for crawler in [
            # PredictorBasedCrawler,
            ExponentialDynamicWeightsMultiPredictorCrawler,
            FollowLeaderMABCrawler,
            BetaDistributionMultiPredictorCrawler,
        ]
        for pred in [multipred5]
        for predictor, trainer in [
            # (pred, online),
            (pred, boost),
            ]
    ])

    metric_decls = [
        (OracleBasedMetric, {'oracle': oracle, 'measure': 'size', 'part': 'crawled'}),
    ]

    n_instances = 20

    mcr = SmartCrawlersRunner(graph_names, crawler_decls, metric_decls, budget)
    mcr.run_adaptive(n_instances, max_memory=16)

    # # Show the results
    from running.merger import ResultsMerger
    crm = ResultsMerger(graph_names, crawler_decls, metric_decls, budget, x_lims=(0, budget))
    crm.draw_by_crawler(x_normalize=False, draw_each_instance=True, draw_error=True, scale=8)
    crm.draw_aggregated(aggregator='TC', scale=3, xticks_rotation=90)
    crm.show_plots()


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
    #logging.getLogger().setLevel(logging.INFO)

    # Single
    for i in range(0, 22):
        single_predictor(config_id=i)

    # Multi
    for i in range(0, 22):
        multi_predictor(config_id=i)
