from crawlers.cbasic import Crawler

# Possible Crawler statistics. Check it to avoid misspelling
CRAWLER_STATISTICS = {
    "crawled_nodes",  # sequence of crawled nodes
    "observed_set_size",  # size of observed set

    # For PredictorBasedCrawler
    "target_flags",  # sequence of target flag for crawled nodes
    "seed_estimation",  # sequence of predictor estimations for crawled nodes

    # More specific
    "dynamic_weights",  # for MAB
    "xgb_feature_importances",  # feature_importances from XGB model
    "xgb_cv_in_train",  # XGB prediction quality measured by CV at training
    "xgb_observed_classification",  # XGB prediction quality of observed nodes

    "train_set_size",  # len(Xs) when training predictor
    "train_set_targets_fraction",  # fraction of target nodes in training set for predictor

    "mab_weights",  # MAB weights for crawlers
}


class StatisticsCrawlerHelper(Crawler):
    """ Helper to manage crawler statistics.
    To make it compute statistics list corresponding flags in 'statistics_flags'

    In order to save statistics results to file, StatisticsSaverMetric should be included to metrics
    declarations.
    """
    def __init__(self, graph, statistics_flags=None, **kwargs):
        """
        :param graph:
        :param statistics_flags: these statistics will be computed together with StatisticsSaverMetric
        parameters. StatisticsSaverMetric is needed for saving their results to file.
        :param kwargs:
        """
        # 'statistics_flags' is not passed to superclass
        super(StatisticsCrawlerHelper, self).__init__(graph=graph, **kwargs)

        # Statistics flags and values
        self._statistics_flags = set()  # set of flags
        self._statistics_values = {}  # statistics_name -> value

        # Set statistics flags
        if isinstance(statistics_flags, str):
            assert statistics_flags in CRAWLER_STATISTICS
            self._statistics_flags.add(statistics_flags)
        elif isinstance(statistics_flags, list):
            for s in statistics_flags:
                assert s in CRAWLER_STATISTICS
                self._statistics_flags.add(s)

        if "seed_estimation" in self._statistics_flags:
            self._statistics_values["seed_estimation"] = []
        if "dynamic_weights" in self._statistics_flags:
            self._statistics_values["dynamic_weights"] = {}
        if "crawled_nodes" in self._statistics_flags:
            self._statistics_values["crawled_nodes"] = []
        if "target_flags" in self._statistics_flags:
            self._statistics_values["target_flags"] = []

        # FIXME looks bad
        if graph is None:
            return

        patch_train_stats = [s for s in [
            "xgb_cv_in_train", "xgb_observed_classification", "train_set_size",
            "train_set_targets_fraction"]
                              if s in self._statistics_flags]

        if len(patch_train_stats) > 0:
            # Patch predictor train method
            try:
                predictor = self.predictor
                orig_train = predictor.train
                for s in patch_train_stats:
                    self._statistics_values[s] = []

                from sklearn.model_selection import cross_val_score

                class patched_train:
                    def __init__(self, crawler, method, counter):
                        self.crawler = crawler
                        self.method = method
                        self.counter = counter

                    def __call__(self, Xs, ys, *args, **kwargs):
                        # Call original train()
                        res = self.method(Xs, ys, *args, **kwargs)

                        if "xgb_cv_in_train" in patch_train_stats:
                            # Measure CV
                            scores = cross_val_score(predictor.model, Xs, ys, cv=5, scoring='f1_macro')
                            mean = scores.mean()
                            # std = scores.std()
                            self.counter["xgb_cv_in_train"].append(mean)

                        if "xgb_observed_classification" in patch_train_stats:
                            # Measure classification quality of observed nodes
                            crawler = self.crawler
                            _Xs = []
                            _ys = []
                            for node in crawler.observed_set:
                                if len(self.crawler.crawled_set) == 0:
                                    continue
                                X = crawler.predictor.extract_features(node, self.crawler)
                                y = 1 if crawler.oracle(node, crawler._orig_graph) else 0
                                _Xs.append(X)
                                _ys.append(y)
                            scores = cross_val_score(crawler.predictor.model, _Xs, _ys, cv=5, scoring='f1_macro')
                            self.counter["xgb_observed_classification"].append(scores.mean())

                        if "train_set_size" in patch_train_stats:
                            self.counter["train_set_size"].append(len(Xs))

                        if "train_set_targets_fraction" in patch_train_stats:
                            # Fraction of 1s in ys
                            fraction = sum(ys) / len(ys)
                            self.counter["train_set_targets_fraction"].append(fraction)

                        return res

                predictor.train = patched_train(
                    self, orig_train, self._statistics_values)
            except AttributeError: pass

    def crawl(self, seed):
        # Count statistics before crawl()
        if "seed_estimation" in self._statistics_flags:
            try:
                self._statistics_values["seed_estimation"].append(self._obs_estimation[seed])
            except AttributeError: pass

        if "dynamic_weights" in self._statistics_flags:
            stat = self._statistics_values["dynamic_weights"]
            if "predict" not in stat:
                stat["predict"] = []
            stat["predict"].append(self.__estimate_node_adv(seed))

            # Dump estimations for all observed_set every 1000 steps
            if len(self.crawled_set) % 100 == 0:
                stat["observed_nodes_estimations"] = {}
                for node in self.observed_set:
                    # TODO try except
                    t = self.oracle(node, self.orig_graph)
                    if t == 1:
                        stat["observed_nodes_estimations"][node] = [True, self._obs_estimation[node]]
                    elif t == 0:
                        stat["observed_nodes_estimations"][node] = [False, self._obs_estimation[node]]

        # Crawl
        res = super(StatisticsCrawlerHelper, self).crawl(seed)

        # Statistics after crawl()
        if "crawled_nodes" in self._statistics_flags:
            self._statistics_values["crawled_nodes"].append(seed)

        if "target_flags" in self._statistics_flags:
            # TODO try except
            self._statistics_values["target_flags"].append(self.oracle(seed, self.orig_graph))

        return res

    def __estimate_node_adv(self, node) -> list:
        """ FIXME should not be here
        """
        neighborhood = self.get_neighborhood(node)
        try:
            len_models_list = len(self.predictor.models_list())
        except AttributeError:
            return []

        if len(neighborhood.neighs) == 0:  # the very 1st node
            return ([0.5] * len_models_list).copy()

        # NOTE: this is time-consuming
        X = self.predictor.extract_features(neighborhood, index=len(self._crawled_set),
                                            graph=self._orig_graph, oracle=self.oracle)
        try:
            return self.predictor.probabilities_adv(X, self._seen_Xs, self._seen_ys).copy()
        except AttributeError: pass

    def collect_statistics(self, **kwargs):
        """ Returns a dictionary composed of statistics computed according to the flags.
        If dictionary is empty, returns None.

        :return: dict of statistics or None if didn't collect any statistics
        """
        statistics_dict = {}

        # Add collected statistics to results if flag was set
        if "crawled_nodes" in self._statistics_flags:
            statistics_dict["crawled_nodes"] = self._statistics_values["crawled_nodes"].copy()
            self._statistics_values["crawled_nodes"].clear()

        if "observed_set_size" in self._statistics_flags:
            statistics_dict["observed_set_size"] = len(self.observed_set)

        if "seed_estimation" in self._statistics_flags:
            statistics_dict["seed_estimation"] = self._statistics_values["seed_estimation"].copy()
            self._statistics_values["seed_estimation"].clear()

        if "target_flags" in self._statistics_flags:
            statistics_dict["target_flags"] = self._statistics_values["target_flags"].copy()
            self._statistics_values["target_flags"].clear()

        if "dynamic_weights" in self._statistics_flags:
            self_stat = self._statistics_values["dynamic_weights"]
            stat = statistics_dict["dynamic_weights"] = {}
            try:
                stat["weights"] = self.predictor.weights()
            except AttributeError: pass

            stat["predict"] = self_stat["predict"].copy()
            self_stat["predict"].clear()

            stat["observed_nodes_estimations"] = self_stat["observed_nodes_estimations"].copy()
            self_stat["observed_nodes_estimations"].clear()

        if "avg_time_per_get_neighborhood" in self._statistics_flags:
            s = self._statistics_values["avg_time_per_get_neighborhood"]
            avg = s["timer"] / s["counter"] if s["counter"] > 0 else 0
            statistics_dict["avg_time_per_get_neighborhood"] = avg
            s["counter"] = 0
            s["timer"] = 0

        if "xgb_feature_importances" in self._statistics_flags:
            imp = []
            try:
                imp = list(self.predictor.model.feature_importances_)
            except AttributeError: pass
            statistics_dict["xgb_feature_importances"] = imp

        if "rf_feature_importances" in self._statistics_flags:
            # TODO
            # https://blog.datadive.net/random-forest-interpretation-with-scikit-learn/
            # from treeinterpreter import treeinterpreter as ti
            pass

        if "mab_weights" in self._statistics_flags:
            weights = []
            try:
                weights = list(self._weights)
            except AttributeError: pass
            statistics_dict["mab_weights"] = weights

        for s in [
            "xgb_cv_in_train", "xgb_observed_classification", "train_set_size",
            "train_set_targets_fraction"]:
            if s in self._statistics_flags:
                statistics_dict[s] = self._statistics_values[s].copy()
                self._statistics_values[s].clear()

        return statistics_dict if len(statistics_dict) > 0 else None
