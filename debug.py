import os
import pandas as pd
from cell import graph_utils
import cell.BiasedDirectedWeightedWalk as BDWW
from stellargraph import StellarDiGraph

layer = 'base_unnormalized_allcombined'

edge_path="/Users/fahimehb/Documents/NPP_GNN_project/dat/edgelists/VISp/"
edge_filename="selfconnection_added_edges_node21_32_removed.csv"

file_name = os.path.join(edge_path, layer, edge_filename)
tmp_edge = pd.read_csv(file_name, index_col="Unnamed: 0")
tmp_edge[['source', 'target']] = tmp_edge[['source', 'target']].astype(str)
nxg = graph_utils.build_nx_graph(source_target_weight=tmp_edge, directed=True)
sdg = StellarDiGraph(nxg)
BDWW.BeginWalk(sdg, begin_checks=True, weighted=True, directed=True)
rw = BDWW.BiasedDirectedRandomWalk(sdg,
                                   directed=True,
                                   weighted=True,
                                   begin_checks=False)

nodes = list(sdg.nodes())
walks = rw.run(nodes=nodes,
               length=2,
               n=1,
               p=1,
               q=1,
               weighted=True,
               directed=True)
print(walks)