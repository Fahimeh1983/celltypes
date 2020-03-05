# How to run
#python -m parallel_biased_directed_walk --edge_path "/home/pogo/work_dir/NPP-GNN-project/dat/graphs/VISp/"
# --output_path "/home/pogo/work_dir/NPP-GNN-project/dat/graphs/VISp/"
# --n 10
# --length 10
# --p 1
# --q 1
# --is_weighted 1
# --is_directed 1

import os
import csv
import argparse
import timeit
import datetime

import pandas as pd
import cell.BiasedDirectedWeightedWalk as BDWW

from cell import graph_utils
from cell import utils
from stellargraph import StellarDiGraph

parser = argparse.ArgumentParser()
parser.add_argument("--edge_path",
                    default="/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/NPP-GNN-project/dat/"
                            "graphs/VISp/",
                    type=str,
                    help="in this dir, there are multiple folders with layer names and in each of them, there is"
                         " a a file called edges.csv")
parser.add_argument("--output_path",
                    default="/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/NPP-GNN-project/dat/"
                            "graphs/VISp/",
                    type=str,
                    help="where to save the walks")
parser.add_argument("--edge_filename",
                    default="myfile.csv",
                    type=str,
                    help="the file name which is in the edge path dir and should be read for the edges")
parser.add_argument("--walk_filename",
                    default="walk.csv",
                    type=str,
                    help="the file name to be used for the output of walk")
parser.add_argument("--n", default=1, type=int, help="number of walks per node")
parser.add_argument("--length", default=10, type=int, help="length of each walk")
parser.add_argument("--p", default=1, type=int, help="p")
parser.add_argument("--q", default=1, type=int, help="q")
parser.add_argument("--is_weighted", default=0, type=int, help="If the graph is weighted")
parser.add_argument("--is_directed", default=0, type=int, help="If the graph is directed")
parser.add_argument("--job_id", default=None, type=int, help="will modify the output walk file name")



def main(edge_path, edge_filename, output_path, walk_filename, n, length, p, q, is_weighted, is_directed, job_id):
    
    start_time = timeit.default_timer()
    if is_weighted == 1:
        weighted = True
    else:
        weighted = False

    if is_directed == 1:
        directed = True
    else:
        directed = False

    layers = os.listdir(edge_path)
    nx_graphs = {}  # keep all the nxDigraphs
    stellar_Di_graphs = {}  # keep all the stellarDigraphs
    node_importance = {}  # keep all the node_importance per layers

    for layer in layers:
        file_name = os.path.join(edge_path, layer, edge_filename)
        tmp_edge = pd.read_csv(file_name, index_col="Unnamed: 0")
        tmp_edge[['source', 'target']] = tmp_edge[['source', 'target']].astype(str)

        # 1) for each layer first create a nx-Digraph
        nxg = graph_utils.Build_nx_Graph(source_target_weight=tmp_edge, directed=True)
        nx_graphs[layer] = nxg

        # 2) Create stellar Di graphs
        sdg = StellarDiGraph(nxg)
        stellar_Di_graphs[layer] = sdg

        # 3) Initialize the walk and return the layer_node_importance
        obj = BDWW.BeginWalk(sdg, begin_checks=True, weighted=True, directed=True)
        node_importance[layer] = obj.node_importance
        print("for layer: ", layer, "this is the end node:", obj.end_nodes)

    # 4) Find all the nodes in all the graphs
    base_nodes = BDWW.get_all_nodes(stellar_Di_graphs)

    # 5) Find the node importance
    layer_importance = BDWW.get_layer_importance(base_nodes, node_importance)

    #6) finally lets walk
    walks = BDWW.biased_directed_multi_walk(stellar_multi_graph_dict=stellar_Di_graphs,
                                            nodes=base_nodes,
                                            layer_importance=layer_importance,
                                            n=n,
                                            length=length,
                                            p=p,
                                            q=q,
                                            tol=10 ** -6,
                                            weighted=weighted,
                                            directed=directed)[0]

    if job_id is not None:
        walk_file_name = str.split(walk_filename, ".")[0] + "_" + str(job_id) + ".csv"
    else:
        walk_file_name = walk_filename
    utils.Write_List_of_Lists_from_CSV(output_path, walk_file_name, walks)

    elapsed = timeit.default_timer() - start_time

    print('-------------------------------')
    print('Training time:', elapsed)
    print('-------------------------------')

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
