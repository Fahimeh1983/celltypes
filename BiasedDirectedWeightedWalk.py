import warnings

import numpy as np

from stellargraph import StellarDiGraph, StellarGraph
from stellargraph.core.utils import is_real_iterable
from stellargraph.random import random_state

class BeginWalk(object):
    """
    Base class for exploring graphs.
    """

    def __init__(self, graph, weighted, directed, begin_checks, seed=None):
        self.graph = graph

        # Initialize the random state
        self._check_seed(seed)

        self._random_state, self._np_random_state = random_state(seed)

        # We require a StellarGraph for this
        if not isinstance(graph, StellarGraph):
            raise TypeError("Graph must be a StellarGraph or StellarDiGraph.")

        if begin_checks:
            self.end_nodes, self.node_importance = self.begin_walk_check(directed, weighted)



    def begin_walk_check(self, directed, weighted):
        if weighted:
            self.check_wts(weighted)

        end_nodes = self.get_end_nodes(directed=directed, weighted=weighted)

        if end_nodes is not None:
            raise TypeError(end_nodes, "Each node must have at least one connection (we had"
                            "to add a self connection to it if there is not any out "
                            "going connection).")

        node_importance = self.get_node_importance()

        return end_nodes, node_importance


    def get_end_nodes(self, directed, weighted):
        """
        It take a graph and if a node does not have any outgoing connection
        then it will add a self connection to it with a very small weight if it is
        weighted.

        Returns
        -------

        """
        if not directed:
            self._raise_error("Not implemented for non directional graphs")

        if not weighted:
            self._raise_error("Not implemented for non weighted graphs")

        for node in self.graph.nodes():
            outgoing_n = self.out_nodes(node)
            if len(outgoing_n) == 0:
                return node


    def get_node_importance(self):
        """
        Compute the importance of each node in each layer
        """
        print("Computing the node importance!")
        node_importance = {}
        for node in self.graph.nodes():
            outgoing_n = self.out_nodes(node)
            outgoing_w = 0
            for o_n in outgoing_n:
                outgoing_w += self.graph._edge_weights(node, o_n)[0]
            node_importance[node] = outgoing_w

        total = sum(node_importance.values(), 0.0)
        node_importance = {k: v / total for k, v in node_importance.items()}

        return node_importance


    def check_wts(self, weighted):
        print("Checking all the weights on all the edges!")
        if weighted:
            # Check that all edge weights are greater than or equal to 0.
            # Also, if the given graph is a MultiGraph, then check that there are no two edges between
            # the same two nodes with different weights.
            for node in self.graph.nodes():
                # TODO Encapsulate edge weights
                for out_neighbor in self.graph.out_nodes(node):

                    wts = set()
                    for weight in self.graph._edge_weights(node, out_neighbor):
                        if weight is None or np.isnan(weight) or weight == np.inf:
                            self._raise_error(
                                "Missing or invalid edge weight ({}) between ({}) and ({}).".format(
                                    weight, node, out_neighbor
                                )
                            )
                        if not isinstance(weight, (int, float)):
                            self._raise_error(
                                "Edge weight between nodes ({}) and ({}) is not numeric ({}).".format(
                                    node, out_neighbor, weight
                                )
                            )
                        if weight < 0:  # check if edge has a negative weight
                            self._raise_error(
                                "An edge weight between nodes ({}) and ({}) is negative ({}).".format(
                                    node, out_neighbor, weight
                                )
                            )

                        wts.add(weight)
                    if len(wts) > 1:
                        # multigraph with different weights on edges between same pair of nodes
                        self._raise_error(
                            "({}) and ({}) have multiple edges with weights ({}). Ambiguous to choose an edge for the random walk.".format(
                                node, out_neighbor, list(wts)
                            )
                        )
        else:
            self._raise_error(
                "This check is done for only weighted graphs."
                )



    def _check_seed(self, seed):
        if seed is not None:
            if type(seed) != int:
                self._raise_error(
                    "The random number generator seed value, seed, should be integer type or None."
                )
            if seed < 0:
                self._raise_error(
                    "The random number generator seed value, seed, should be non-negative integer or None."
                )

    def _get_random_state(self, seed):
        """
        Args:
            seed: The optional seed value for a given run.

        Returns:
            The random state as determined by the seed.
        """
        if seed is None:
            # Restore the random state
            return self._random_state
        # seed the random number generator
        rs, _ = random_state(seed)
        return rs

    def neighbors(self, node):
        if not self.graph.has_node(node):
            self._raise_error("node {} not in graph".format(node))
        return self.graph.neighbors(node)

    def out_nodes(self, node):
        if not self.graph.has_node(node):
            self._raise_error("node {} not in graph".format(node))
        return self.graph.out_nodes(node)

    def in_nodes(self, node):
        if not self.graph.has_node(node):
            self._raise_error("node {} not in graph".format(node))
        return self.graph.in_nodes(node)

    def run(self, *args, **kwargs):
        """
        To be overridden by subclasses. It is the main entry point for performing random walks on the given
        graph.

        It should return the sequences of nodes in each random walk.
        """
        raise NotImplementedError

    def _raise_error(self, msg):
        raise ValueError("({}) {}".format(type(self).__name__, msg))

    def _check_common_parameters(self, nodes, n, length, seed):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            nodes: <list> A list of root node ids from which to commence the random walks.
            n: <int> Number of walks per node id.
            length: <int> Maximum length of each walk.
            seed: <int> Random number generator seed.
        """
        self._check_nodes(nodes)
        self._check_repetitions(n)
        self._check_length(length)
        self._check_seed(seed)

    def _check_nodes(self, nodes):
        if nodes is None:
            self._raise_error("A list of root node IDs was not provided.")
        if not is_real_iterable(nodes):
            self._raise_error("Nodes parameter should be an iterable of node IDs.")
        if (
            len(nodes) == 0
        ):  # this is not an error but maybe a warning should be printed to inform the caller
            warnings.warn(
                "No root node IDs given. An empty list will be returned as a result.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _check_repetitions(self, n):
        if type(n) != int:
            self._raise_error(
                "The number of walks per root node, n, should be integer type."
            )
        if n <= 0:
            self._raise_error(
                "The number of walks per root node, n, should be a positive integer."
            )

    def _check_length(self, length):
        if type(length) != int:
            self._raise_error("The walk length, length, should be integer type.")
        if length <= 0:
            # Technically, length 0 should be okay, but by consensus is invalid.
            self._raise_error("The walk length, length, should be a positive integer.")

    # For neighbourhood sampling
    def _check_sizes(self, n_size):
        err_msg = "The neighbourhood size must be a list of non-negative integers."
        if not isinstance(n_size, list):
            self._raise_error(err_msg)
        if len(n_size) == 0:
            # Technically, length 0 should be okay, but by consensus it is invalid.
            self._raise_error("The neighbourhood size list should not be empty.")
        for d in n_size:
            if type(d) != int or d < 0:
                self._raise_error(err_msg)


def get_all_nodes(dict_of_all_graphs):
    """
    Takes a lot og graphs and find all the nodes corresponding to al the graphs together
    """
    base_nodes = []
    for k, v in dict_of_all_graphs.items():
        print("number of nodes for layer: ", k, " is:", len(v.nodes()))
        base_nodes = base_nodes + list(v.nodes())

    base_nodes = list(set(base_nodes))
    return base_nodes


def get_layer_importance(nodes, node_importance):
    """
    Takes all the node importance and all the nodes and return the layer
    importance. It is nothing but rearranging the dicts.
    It takes a dict which has the layers as keys and for each key, it has
    a dict and the keys of those dicts are node and the values are the importance.

    This function will rearrange the given dict and make another dict which has nodes as keys
    and the values are dicts that have layers as keys and values as importance
    """
    layer_importance = {}
    for node in nodes:
        layer_importance[node] = {}
        for k, v in node_importance.items():
            if node in v:
                layer_importance[node][k] = v[node]

    for key, val in layer_importance.items():
        total = sum(val.values(), 0.0)
        if total != 0:
            layer_importance[key] = {k: v / total for k, v in layer_importance[key].items()}

    return layer_importance


# def get_end_nodes(layer_importance):
#     """
#     Gets the layer importance dict and return the nodes that don't have any outgoing
#     edge in any layer
#     """
#
#     end_node = []
#     for k, v in layer_importance.items():
#         if sum(v.values()) == 0:
#             end_node.append(k)
#
#     return end_node



def naive_weighted_choices(rs, weights):
    """
    Select an index at random, weighted by the iterator `weights` of
    arbitrary (non-negative) floats. That is, `x` will be returned
    with probability `weights[x]/sum(weights)`.

    For doing a single sample with arbitrary weights, this is much (5x
    or more) faster than numpy.random.choice, because the latter
    requires a lot of preprocessing (normalized probabilties), and
    does a lot of conversions/checks/preprocessing internally.
    """

    # divide the interval [0, sum(weights)) into len(weights)
    # subintervals [x_i, x_{i+1}), where the width x_{i+1} - x_i ==
    # weights[i]
    subinterval_ends = []
    running_total = 0
    for w in weights:
        if w < 0:
            raise ValueError("Detected negative weight: {}".format(w))
        running_total += w
        subinterval_ends.append(running_total)

    # pick a place in the overall interval
    x = rs.random() * running_total

    # find the subinterval that contains the place, by looking for the
    # first subinterval where the end is (strictly) after it
    for idx, end in enumerate(subinterval_ends):
        if x < end:
            break

    return idx



class BiasedDirectedRandomWalk(BeginWalk):
    """
    The same as biased random walk, but it will take a directed graph
    """
    def run(self, nodes, n, length, p=1.0, q=1.0, weighted=False, directed=False, seed=None):

        """
        Perform a random walk starting from the root nodes.

        Args:
            nodes (list): The root nodes as a list of node IDs
            n (int): Total number of random walks per root node
            length (int): Maximum length of each random walk
            p (float, default 1.0): Defines probability, 1/p, of returning to source node
            q (float, default 1.0): Defines probability, 1/q, for moving to a node away from the source node
            seed (int, optional): Random number generator seed; default is None
            weighted (bool, default False): Indicates whether the walk is unweighted or weighted
            directed

        Returns:
            List of lists of nodes ids for each of the random walks

        """
        self._check_common_parameters(nodes, n, length, seed)
        self._check_weights(p, q, weighted)
        rs = self._get_random_state(seed)

        if not directed:
            self._raise_error(
                "The graph must be directed")

        if not self.graph.is_directed():
            self._raise_error(
                "You think that you have a directed graph but it is not")

        ip = 1.0 / p
        iq = 1.0 / q

        walks = []
        for node in nodes:  # iterate over root nodes
            print(node)
            for walk_number in range(n):  # generate n walks per root node
                # the walk starts at the root
                walk = [node]

                out_neighbours = self.out_nodes(node)

                previous_node = node
                previous_node_neighbours = out_neighbours

                # calculate the appropriate un-normalised transition
                # probability, given the history of the walk
                def transition_probability(nn, current_node, weighted):

                    if weighted:
                        # TODO Encapsulate edge weights
                        weight_cn = self.graph._edge_weights(current_node, nn)[0]
                    else:
                        weight_cn = 1.0

                    if nn == previous_node:  # d_tx = 0
                        # print(nn, "I am here0")
                        return ip * weight_cn
                    elif nn in previous_node_neighbours:  # d_tx = 1
                        # print(nn, "I am here1")
                        return 1.0 * weight_cn
                    else:  # d_tx = 2
                        # print(nn, "I am here2")
                        return iq * weight_cn

                if out_neighbours:

                    neighbors_weight_list = [self.graph._edge_weights(previous_node, out_n)[0] for out_n in out_neighbours]
                    # print(out_neighbours, neighbors_weight_list)
                    current_node = np.random.choice(out_neighbours, 1, p=neighbors_weight_list)[0]
                    # current_node = rs.choice(out_neighbours)
                    for _ in range(length - 1):
                        walk.append(current_node)
                        out_neighbours = self.out_nodes(current_node)

                        if not out_neighbours:
                            print(node)
                            raise ValueError("Every node should at least have one outgoing"
                                             "edge from itself to itself")
                            break

                        # select one of the neighbours using the
                        # appropriate transition probabilities
                        choice = naive_weighted_choices(
                            rs,
                            (
                                transition_probability(nn, current_node, weighted)
                                for nn in out_neighbours
                            ),
                        )

                        previous_node = current_node
                        previous_node_neighbours = out_neighbours
                        current_node = out_neighbours[choice]

                walks.append(walk)

        return walks

    def _check_weights(self, p, q, weighted):
        """
        Checks that the parameter values are valid or raises ValueError exceptions with a message indicating the
        parameter (the first one encountered in the checks) with invalid value.

        Args:
            p: <float> The backward walk 'penalty' factor.
            q: <float> The forward walk 'penalty' factor.
            weighted: <False or True> Indicates whether the walk is unweighted or weighted.
       """
        if p <= 0.0:
            self._raise_error("Parameter p should be greater than 0.")

        if q <= 0.0:
            self._raise_error("Parameter q should be greater than 0.")

        if type(weighted) != bool:
            self._raise_error(
                "Parameter weighted has to be either False (unweighted random walks) or True (weighted random walks)."
            )




def biased_directed_multi_walk(stellar_multi_graph_dict, nodes, layer_importance, n=10, length=10, p=1.0, q=1.0,
                               tol=10**-6, weighted=True, directed=True):
    """

    Parameters
    ----------
    stellar_multi_graph_dict: This is a dictionary of all the graphs. The keys are the layer names and the values
    are stellar graph objects for that layer.
    nodes: All possible the nodes in all the graphs, it is a unique list, no repeats
    layer_importance: it is a dictionary which as nodes as keys and the values are another dictionary which has
    the layers importance for that node
    n: number of walks per node
    length: length of each walk
    p: similar to node2vec
    q: similar to node2vec
    weighted
    directed

    Returns
    -------
    list of walks and list of layer jumps during those walks

    """
    walks = []
    layers = []

    for node in nodes:
        walk_per_node = 0
        while walk_per_node < n:
            current_node = node
            walk = [node]
            layer = []

            while len(walk) < length:
                choice_list = list(layer_importance[current_node].keys())
                choice_p = list(layer_importance[current_node].values())
                if abs(sum(choice_p) - 1) > tol:
                    raise ValueError("The sum of probabilities for node: ", node, "is: ", sum(choice_p))
                else:
                    select_layer = np.random.choice(choice_list, size=1, replace=True, p=choice_p)[0]
                    layer.append(select_layer)

                previous_node = current_node

                rw = BiasedDirectedRandomWalk(stellar_multi_graph_dict[select_layer],
                                              directed=directed,
                                              weighted=weighted,
                                              begin_checks=False)

                current_node = rw.run(nodes=[previous_node],
                                      length=2,
                                      n=1,
                                      p=p,
                                      q=q,
                                      weighted=weighted,
                                      directed=directed)[0][1]

                walk.append(current_node)

            walk_per_node += 1
            walks.append(walk)
            layers.append(layer)

    return walks, layers


# from cell import utils, graph_utils
# import pandas as pd
#
#
# npp_adj = np.zeros((93,93))
# layers = utils.get_npp_visp_layers()
#
# for layer in layers:
#     path = utils.get_npp_visp_interaction_mat_path(layer)
#     tmp_inter= pd.read_csv(path, index_col="Unnamed: 0")
#     npp_adj = npp_adj + tmp_inter.values
#
# npp_adj = pd.DataFrame(npp_adj)
# npp_adj.index = npp_adj.index.astype(str)
# npp_adj.columns = npp_adj.columns.astype(str)
#
# npp_adj = npp_adj.drop('21', axis=0)
# npp_adj = npp_adj.drop('21', axis=1)
# npp_adj = npp_adj.drop('32', axis=0)
# npp_adj = npp_adj.drop('32', axis=1)
#
# edges = graph_utils.build_edge_list(weight_matrix=npp_adj, threshold=None, directed=True)
#
# nxg = graph_utils.build_nx_graph(source_target_weight=edges, directed=True)
#
# # 2) Create stellar Di graphs
# sdg = StellarDiGraph(nxg)
# BeginWalk(sdg, begin_checks=True, weighted=True, directed=True)
# rw = BiasedDirectedRandomWalk(sdg, directed=True, weighted=True, begin_checks=False)
#
# nodes = list(sdg.nodes())
# walks = rw.run(nodes=nodes, length=100, n=100, p=1, q=1, weighted=True, directed=True)