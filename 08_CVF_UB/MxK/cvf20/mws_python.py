import numpy as np

from .unionfind import UnionFind


# use these functions to keep track of all active mutual exclusion constraints
def check_mutex(M, root_i, root_j):
    return len(M[root_i].intersection(M[root_j])) != 0


def merge_mutex(M, new_root, root_i, root_j):
    M[new_root] = M[root_i].union(M[root_j])


def add_mutex(M, root_i, root_j, idx):
    M[root_i].add(idx)
    M[root_j].add(idx)


def MWS(graph, edges, edge_costs):

    nodes = np.unique(edges)
    uf = UnionFind(elements=nodes)
    # Initialize set of constraints:
    M = dict((n, set()) for n in nodes)

    for idx in np.argsort(-np.abs(edge_costs), axis=None):
        e = edges[idx]
        root_i = uf[uf.find(e[0])]
        root_j = uf[uf.find(e[1])]
        if edge_costs[idx] > 0:
            if not check_mutex(M, root_i, root_j):
                if not uf.connected(e[0], e[1]):
                    uf.union(e[0], e[1])
                    new_root = uf[uf.find(e[0])]
                    merge_mutex(M, new_root, root_i, root_j)
        else:
            if not uf.connected(e[0], e[1]):
                add_mutex(M, root_i, root_j, idx)

    # this will export the labeling obtained by the MWS
    return np.asarray([0] + [uf.find(x) for x in nodes])

