from experiments.paper_experiments import paper_configs
from running.metrics import OracleBasedMetric
from search.feature_extractors import NeighborsFeatureExtractor
from search.oracles import HasAttrValueOracle
from search.predictor_based_crawlers.mab import ExponentialDynamicWeightsMultiPredictorCrawler, \
    FollowLeaderMABCrawler, BetaDistributionMultiPredictorCrawler, MultiPredictor
from search.predictor_based_crawlers.predictor_based import PredictorBasedCrawler
from search.predictor_based_crawlers.training_strategies import BoostingTrainStrategy
from search.predictors.simple_predictors import MaximumTargetNeighborsPredictor, SklearnPredictor


def big_table(mode='tex', out='result'):
    assert mode in ['tex', 'excel']
    assert out in ['result', 'runs']

    from running.merger import ResultsMerger
    from search.predictors.gnn_predictors import GNNPredictor
    # Graph configs
    configs = [
        0,
        1,
        2,
        3,
        4, 5, 6,
        7, 8, 9, 10, 11, 12, 13,
        14,
        15, 16, 17, 18, 19, 20, 21
    ]

    table_rows = []
    for i in configs:
        config = paper_configs[i]
        graph_names = [config[0]]
        attr = config[1]
        value = config[2]
        budget = config[3]
        title = config[4]
        oracle = HasAttrValueOracle(attribute=attr, value=value)

        mtn = MaximumTargetNeighborsPredictor(name='MTN')
        nfe = NeighborsFeatureExtractor(
            od=True, cc=True, cnf=True, tnf=True, tri=True, neighs1=True, hist=5)
        xgb = SklearnPredictor("GradientBoostingClassifier", nfe, name=f"XGB")
        rf = SklearnPredictor("RandomForestClassifier", nfe, n_estimators=100, name=f"RF")
        knn = SklearnPredictor("KNeighborsClassifier", nfe, algorithm='kd_tree', n_neighbors=30, name="KNN")
        svc = SklearnPredictor("SVC", nfe, kernel='rbf', probability=True, name="SVC")
        hidden = 5
        epochs = 200
        gcn = GNNPredictor('GraphConv', (hidden,), attributes=[], epochs=epochs, name='GCN')
        sage_gcn = GNNPredictor('SAGEConv', (hidden,), aggregator_type='gcn',
                                attributes=[], epochs=epochs, name="SAGE")
        gat1 = GNNPredictor('GATConv', (hidden,), num_heads=1, merge='mean',
                            attributes=[], epochs=epochs, name="GAT-1")
        gat3 = GNNPredictor('GATConv', (hidden,), num_heads=3, merge='mean',
                            attributes=[], epochs=epochs, name="GAT-3")
        multipred5 = MultiPredictor([mtn, xgb, knn, rf, svc], name="Multi-5")
        multipred6 = MultiPredictor([mtn, xgb, knn, rf, svc, sage_gcn], name="Multi-6")

        boost = BoostingTrainStrategy(name="boost")

        # Crawler configs
        crawler_decls = []
        crawler_decls.extend([
            # (RandomCrawler, {'oracle': oracle, 'initial_seed': 'target', 'name': 'RC'}),
            (PredictorBasedCrawler, {
                'predictor': mtn.declaration, 'oracle': oracle, 'initial_seed': 'target', 'name': "MTN",
            }),
        ])
        crawler_decls.extend([
            (PredictorBasedCrawler, {
                'predictor': predictor.declaration, 'training_strategy': boost.declaration,
                're_estimate': 'after_train', 'initial_seed': 'target', 'oracle': oracle,
                'name': predictor.name,
            })
            for predictor in [
                xgb, rf, knn, svc,
                gcn, sage_gcn, gat1, gat3
            ]
        ])
        crawler_decls.extend([
            (crawler, {
                'predictor': predictor.declaration, 'training_strategy': boost.declaration,
                're_estimate': 'after_train', 'initial_seed': 'target', 'oracle': oracle,
                'name': cn + "-" + pn,
            })
            for cn, crawler in [
                ("DW", ExponentialDynamicWeightsMultiPredictorCrawler),
                ("FL", FollowLeaderMABCrawler),
                ("Beta", BetaDistributionMultiPredictorCrawler),
            ]
            for pn, predictor in [
                ("5", multipred5),
            ]
        ])
        crawler_decls.extend([
            (ExponentialDynamicWeightsMultiPredictorCrawler, {
                'predictor': multipred6.declaration, 'training_strategy': boost.declaration,
                're_estimate': 'after_train', 'initial_seed': 'target', 'oracle': oracle,
                'name': "DW-6",
            }),
            (FollowLeaderMABCrawler, {
                'predictor': multipred6.declaration, 'training_strategy': boost.declaration,
                're_estimate': 'after_train', 'initial_seed': 'target', 'oracle': oracle,
                'name': "FL-6",
            }),
        ])
        metric_decls = [
            (OracleBasedMetric, {'oracle': oracle, 'measure': 'size', 'part': 'crawled'}),
        ]
        crm = ResultsMerger(graph_names, crawler_decls, metric_decls, budget, x_lims=(0, budget))

        # crm.move_folders(path_to="/home/misha/workspace/crawling/results_ordered", copy=False)

        str_res = crm.get_aggregated(aggregator='TC', print_results=True)

        # First row
        if len(table_rows) == 0:
            if mode == 'tex':
                table_rows.append(
                    "Graph  & " +
                    " & ".join(["\\textbf{%s}" % kw['name'] for _, kw in crawler_decls]) + " \\\\ \\hline")
            elif mode == 'excel':
                table_rows.append("\t".join(["Graph"] + [kw['name'] for _, kw in crawler_decls]))

        # Content row
        # table_row = ["\\begin{tabular}{@{}c@{}}%s\\\\(%s)\\end{tabular}" % (title, budget)]
        table_row = [title]
        for num_instances, g, c, m, mean, var in str_res:
            if out == 'result':
                if mode == 'excel':
                    table_row.append("%.1f" % (mean))
                    # table_row.append("%.1f±%.1f" % (mean, var))
                elif mode == 'tex':
                    table_row.append("%.f$\pm$%.f" % (mean, var))
            elif out == 'runs':
                table_row.append(str(num_instances))
        if mode == 'tex':
            table_row = " & ".join(table_row) + " \\\\"
        elif mode == 'excel':
            table_row = '\t'.join(table_row)
        table_rows.append(table_row)

    for row in table_rows:
        print(row)


if __name__ == '__main__':
    # # All results, Table 4
    big_table(mode='excel', out='result')
    # big_table(mode='excel', out='runs')

