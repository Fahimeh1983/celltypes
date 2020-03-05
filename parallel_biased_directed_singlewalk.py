# How to run
#python -m parallel_biased_directed_walk --edge_path "/home/pogo/work_dir/NPP-GNN-project/dat/graphs/VISp/"
# --output_path "/home/pogo/work_dir/NPP-GNN-project/dat/graphs/VISp/"
# --n 10
# --length 10
# --p 1
# --q 1
# --is_weighted 1
# --is_directed 1

import argparse
import os

import pandas as pd

from stellargraph import StellarDiGraph
from cell import graph_utils, utils
import cell.BiasedDirectedWeightedWalk as BDWW

parser = argparse.ArgumentParser()
parser.add_argument("--layer",
                    default=None,
                    type=str,
                    help="This is the name of the folder of the edges or the name of the graph")
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


def main(layer, edge_path, edge_filename, output_path, walk_filename, n, length, p, q, is_weighted, is_directed, job_id):

    if is_weighted == 1:
        weighted = True
    else:
        weighted = False

    if is_directed == 1:
        directed = True
    else:
        directed = False

    file_name = os.path.join(edge_path, layer, edge_filename)
    tmp_edge = pd.read_csv(file_name, index_col="Unnamed: 0")
    tmp_edge[['source', 'target']] = tmp_edge[['source', 'target']].astype(str)

    # 1) for each layer first create a nx-Digraph
    nxg = graph_utils.Build_nx_Graph(source_target_weight=tmp_edge, directed=True)

    # 2) Create stellar Di graphs
    sdg = StellarDiGraph(nxg)

    # 3) Initialize the walk and do the begin checks
    BDWW.BeginWalk(sdg, begin_checks=True, weighted=True, directed=True)

    rw = BDWW.BiasedDirectedRandomWalk(sdg,
                                       directed=True,
                                       weighted=True,
                                       begin_checks=False)

    nodes = list(sdg.nodes())
    walks = rw.run(nodes=nodes,
                   length=length,
                   n=n,
                   p=p,
                   q=q,
                   weighted=weighted,
                   directed=directed)

    result_path = os.path.join(output_path, layer)
    if not os.path.isdir(result_path):
        print("making a new directory for the output")
        os.mkdir(result_path)

    if job_id is not None:
        walk_file_name = str.split(walk_filename, ".")[0] + "_" + str(job_id) + ".csv"
    else:
        walk_file_name = walk_filename
    utils.Write_List_of_Lists_from_CSV(result_path, walk_file_name, walks)

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))