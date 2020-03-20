import matplotlib.pyplot as plt
import numpy as np

METRICS_LIST = ['nodes', 'degrees', 'k_cores', 'eccentricity', 'betweenness_centrality']
property_name = {
    'nodes': 'nodes',
    'degrees': 'degrees',
    'k_cores': 'k-cores',
    'eccentricity': 'eccentricity',
    'betweenness_centrality': 'betweenness'
}

graph_names = ['hamsterster', 'DCAM', 'facebook', 'slashdot', 'github', 'dblp2010']
method_names = ['RC', 'RW', 'BFS', 'DFS', 'MOD', 'DE']
PROPERTIES_COLOR = {
    'nodes': 'k',
    'degrees': 'g',
    'k_cores': 'b',
    'eccentricity': 'r',
    'betweenness_centrality': 'y'}

aucs = list(range(6))  # list of graph ids -> dict of (method -> dict of (prop -> auc))

aucs[0] = {'RC': {'degrees': 1228.054268292683, 'k_cores': 1215.906517094017, 'eccentricity': 1188.8954485488127, 'betweenness_centrality': 1188.3904702970299, 'nodes': 1689.56484375}, 'RW': {'degrees': 1654.9246951219511, 'k_cores': 1598.962873931624, 'eccentricity': 1406.3852242744063, 'betweenness_centrality': 1511.6426361386139, 'nodes': 1705.424125}, 'BFS': {'degrees': 1638.7060975609756, 'k_cores': 1554.5769230769233, 'eccentricity': 1452.7612137203166, 'betweenness_centrality': 1511.1862623762377, 'nodes': 1663.4525937499998}, 'DFS': {'degrees': 1712.983536585366, 'k_cores': 1631.6132478632478, 'eccentricity': 1412.9281002638522, 'betweenness_centrality': 1607.7642326732673, 'nodes': 1730.02903125}, 'MOD': {'degrees': 1839.4393292682926, 'k_cores': 1798.7230235042734, 'eccentricity': 1569.957783641161, 'betweenness_centrality': 1634.1878094059407, 'nodes': 1696.5955000000001}, 'DE': {'degrees': 1845.234756097561, 'k_cores': 1675.1490384615383, 'eccentricity': 1585.1256596306068, 'betweenness_centrality': 1669.9814356435643, 'nodes': 1724.8537812499999}}
aucs[1] = {'RC': {'degrees': 1581.9997727272728, 'k_cores': 1575.1044413919412, 'eccentricity': 1430.693375576037, 'betweenness_centrality': 1471.2300649350648, 'nodes': 2397.0032346605067}, 'RW': {'degrees': 2213.5583766233767, 'k_cores': 2151.9999313186813, 'eccentricity': 1626.4318996415768, 'betweenness_centrality': 1904.3986688311688, 'nodes': 2370.620500025955}, 'BFS': {'degrees': 2184.6341233766234, 'k_cores': 2081.6865384615385, 'eccentricity': 1648.4262352790577, 'betweenness_centrality': 1933.4977272727272, 'nodes': 2359.053905601121}, 'DFS': {'degrees': 1969.6974350649352, 'k_cores': 1813.77673992674, 'eccentricity': 1517.0249423963132, 'betweenness_centrality': 1903.8099350649352, 'nodes': 2342.8771250778655}, 'MOD': {'degrees': 2488.0243506493507, 'k_cores': 2491.2644001831504, 'eccentricity': 1704.6773553507426, 'betweenness_centrality': 2040.2287012987013, 'nodes': 2357.6423996833473}, 'DE': {'degrees': 2481.6218506493506, 'k_cores': 2461.8497252747256, 'eccentricity': 1722.3821492575526, 'betweenness_centrality': 2080.280649350649, 'nodes': 2379.445880268895}}
aucs[2] = {'RC': {'degrees': 38621.35896464647, 'k_cores': 38534.83457267305, 'eccentricity': 36580.577431435624, 'betweenness_centrality': 34036.00330809275, 'nodes': 54053.47089735298}, 'RW': {'degrees': 53973.44706682207, 'k_cores': 53160.87283197301, 'eccentricity': 43640.15642669136, 'betweenness_centrality': 35682.27001656413, 'nodes': 54597.70031214506}, 'BFS': {'degrees': 50487.601029526035, 'k_cores': 47896.3199752786, 'eccentricity': 43601.970043705725, 'betweenness_centrality': 35602.668747042124, 'nodes': 53927.85231574962}, 'DFS': {'degrees': 49742.22907925408, 'k_cores': 46404.0798344059, 'eccentricity': 43457.550740885155, 'betweenness_centrality': 35627.9419439186, 'nodes': 54147.209517762494}, 'MOD': {'degrees': 58210.67495143745, 'k_cores': 59441.47750549365, 'eccentricity': 46059.54800318451, 'betweenness_centrality': 36681.14742427828, 'nodes': 52434.49469668885}, 'DE': {'degrees': 58613.60299145299, 'k_cores': 59227.96332993251, 'eccentricity': 45655.576578442844, 'betweenness_centrality': 35619.603952910555, 'nodes': 52762.80464806285}}
aucs[3] = {'RC': {'degrees': 35230.559407169945, 'k_cores': 35055.179265610765, 'eccentricity': 30862.205918742016, 'betweenness_centrality': 30227.27441250352, 'nodes': 38740.84033407815}, 'RW': {'degrees': 43553.8634381692, 'k_cores': 42843.201190661304, 'eccentricity': 33651.46606838777, 'betweenness_centrality': 33082.3710857324, 'nodes': 45010.56963911674}, 'BFS': {'degrees': 42979.61018569881, 'k_cores': 42509.968872870246, 'eccentricity': 39905.00648547254, 'betweenness_centrality': 32480.01788020002, 'nodes': 44500.89289384783}, 'DFS': {'degrees': 45388.438151461596, 'k_cores': 44357.187845702734, 'eccentricity': 29658.696973066046, 'betweenness_centrality': 34046.28715179494, 'nodes': 45889.84331102603}, 'MOD': {'degrees': 48092.01864244434, 'k_cores': 47838.28506730791, 'eccentricity': 32477.138533494497, 'betweenness_centrality': 38261.833404461184, 'nodes': 47249.35857994413}, 'DE': {'degrees': 47724.78343417013, 'k_cores': 47478.17280277451, 'eccentricity': 35041.711880625495, 'betweenness_centrality': 31015.743357860563, 'nodes': 46459.8208631401}}
aucs[4] = {'RC': {'degrees': 84885.53571647462, 'k_cores': 85020.39938941522, 'eccentricity': 75898.16255985906, 'betweenness_centrality': 81133.46893873584, 'nodes': 91760.13780871221}, 'RW': {'degrees': 106137.82999816132, 'k_cores': 105623.66718276773, 'eccentricity': 83537.49082217779, 'betweenness_centrality': 98638.0909499876, 'nodes': 101234.4786161213}, 'BFS': {'degrees': 107717.47261890168, 'k_cores': 108582.08244835754, 'eccentricity': 95193.97637607814, 'betweenness_centrality': 100074.16474828328, 'nodes': 101315.34548566584}, 'DFS': {'degrees': 106644.75857031954, 'k_cores': 103660.80692319637, 'eccentricity': 73568.4074701385, 'betweenness_centrality': 101122.44863282864, 'nodes': 102517.09019877549}, 'MOD': {'degrees': 114025.98770123397, 'k_cores': 114407.69025199968, 'eccentricity': 87881.90735790276, 'betweenness_centrality': 106138.06137792669, 'nodes': 105279.5972226244}, 'DE': {'degrees': 113676.99961694046, 'k_cores': 114261.75848411897, 'eccentricity': 88153.29075007483, 'betweenness_centrality': 103704.32734032431, 'nodes': 103388.48568702686}}
aucs[5] = {'RC': {'degrees': 152718.55034412313, 'k_cores': 143153.68304196178, 'eccentricity': 140778.72848048748, 'betweenness_centrality': 148681.54294373535, 'nodes': 164716.68598700163}, 'RW': {'degrees': 180096.77364482926, 'k_cores': 171825.1182811374, 'eccentricity': 138945.62862328062, 'betweenness_centrality': 166609.6341860619, 'nodes': 170797.1613210703}, 'BFS': {'degrees': 167796.8726446904, 'k_cores': 152101.24831697132, 'eccentricity': 160366.10427081317, 'betweenness_centrality': 166601.41586803872, 'nodes': 166425.06765015703}, 'DFS': {'degrees': 169806.8828915268, 'k_cores': 148695.80089372795, 'eccentricity': 121369.332730938, 'betweenness_centrality': 173576.47393510136, 'nodes': 177608.49894661526}, 'MOD': {'degrees': 194274.42335574597, 'k_cores': 181752.1484616852, 'eccentricity': 145555.67116278416, 'betweenness_centrality': 180998.68079814955, 'nodes': 175025.56380237665}, 'DE': {'degrees': 172328.04028323214, 'k_cores': 129957.19227874132, 'eccentricity': 144602.19879618502, 'betweenness_centrality': 185995.6757883231, 'nodes': 179869.23188901474}}


wins = {
    'RC': dict(zip(METRICS_LIST, [0,0,0,0,0])),
    'RW': dict(zip(METRICS_LIST, [0,0,0,0,0])),
    'DFS': dict(zip(METRICS_LIST, [0,0,0,0,0])),
    'BFS': dict(zip(METRICS_LIST, [0,0,0,0,0])),
    'DE': dict(zip(METRICS_LIST, [0,0,0,0,0])),
    'MOD': dict(zip(METRICS_LIST, [0,0,0,0,0]))
}


# for method in method_names:
for prop in METRICS_LIST:
    d = dict(zip(graph_names, [0,0,0,0,0]))
    for graph in range(6):
        best_method = None
        max = 0
        for method in method_names:
            val = aucs[graph][method][prop]
            if val > max:
                max = val
                best_method = method
        wins[best_method][prop] += 1

print(wins)

lines = []
xs = np.array(list(range(len(method_names))))
prev_bottom = np.array([0] * len(method_names))
for p, prop in enumerate(METRICS_LIST):
    h = []
    w = 0.8
    i = 0
    for method in method_names:
        h.append(wins[method][prop])
        i += 1
    line = plt.bar(xs, h, w, bottom=prev_bottom, color=PROPERTIES_COLOR[prop])
    lines.append(line)
    prev_bottom += h

plt.legend(lines, map(lambda m: property_name[m], METRICS_LIST))
plt.xticks(xs, method_names)
plt.ylabel('leader count')
plt.suptitle('Aggregated')
plt.show()