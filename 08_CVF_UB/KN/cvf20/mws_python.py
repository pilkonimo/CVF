import numpy as np

from .unionfind import UnionFind


# use these functions to keep track of all active mutual exclusion constraints
def check_mutex(M, root_i, root_j):
    return root_i in M[root_j] or root_j in M[root_i]

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

    for idx in np.argsort(np.abs(edge_costs), axis=None):
        e = edges[idx]

        if(edge_costs[idx]) > 0:
            if not check_mutex(M, uf[uf.find(e[0])], uf[uf.find(e[1])]):
                if not uf.connected(e[0], e[1]):
                    new_root = uf.find(e[0])
                    merge_mutex(M, new_root, uf[uf.find(e[0])], uf[uf.find(e[1])])
                    uf.union(e[0], e[1])
        else:
            if not uf.connected(e[0], e[1]):
                add_mutex(M, uf[uf.find(e[0])], uf[uf.find(e[1])], idx)


        #######
        # your implementation should go here
        # below you will find a few function calls that could come in handy
        #######


        # This is how you find the root node in a ufd
        # root_i = uf[uf.find(e[0])]

        # this will merge two nodes e[0] and e[1]
        # uf.union(e[0], e[1])
        # new_root = uf.find(e[0])

        # this is how you can check if two nodes are connected
        # uf.connected(e[0], e[1])

    # this will export the labeling obtained by the MWS
    return np.asarray([0] + [uf.find(x) for x in nodes])
