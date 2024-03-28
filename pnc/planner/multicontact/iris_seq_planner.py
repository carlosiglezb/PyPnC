from itertools import product
from typing import List
import numpy as np
import scipy as sp
import networkx as nx
import cvxpy as cp
from time import time


from vision.iris.iris_geom_interface import IrisGeomInterface


class IrisGraph(nx.Graph):
    def __init__(self, iris_regions: List[IrisGeomInterface], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.iris_regions_lst = iris_regions
        self.iris_intersections = {}
        self.computeLineGraph()

        # Place representative point in each box intersection.
        self.optimize_points()

        # Assign fixed length to each edge of the line graph.
        for e in self.edges:
            pu = self.nodes[e[0]]['point']
            pv = self.nodes[e[1]]['point']
            self.edges[e]['weight'] = np.linalg.norm(pv - pu)

        # Store adjacency matrix for scipy's shortest-path algorithms.
        self.adj_mat = nx.to_scipy_sparse_array(self)

    def computeLineGraph(self):
        """
        Compute line graph using networkx.
        """
        inters_graph = nx.Graph()
        for i, iris_i in enumerate(self.iris_regions_lst):
            self.iris_intersections[i] = set()
            for j, iris_j in enumerate(self.iris_regions_lst):
                if i == j:
                    continue
                iris_intersect = iris_i.iris_region.Intersection(iris_j.iris_region,
                                                                    check_for_redundancy=True)
                if not iris_intersect.IsEmpty():
                    self.iris_intersections[i].add(j)
        inters_graph.add_nodes_from(self.iris_intersections.keys())
        for k, k_inters in self.iris_intersections.items():
            k_inters_unique = [l for l in k_inters if l > k]
            inters_graph.add_edges_from(product([k], k_inters_unique))
        line_graph = nx.line_graph(inters_graph)
        self.add_nodes_from(line_graph.nodes)
        self.add_edges_from(line_graph.edges)
        self.v2i = {v: i for i, v in enumerate(self.nodes)}
        self.i2v = {i: v for i, v in enumerate(self.nodes)}

        # Pair each vertex with the corresponding intersection.
        for v in self.nodes:
            iris_k = self.iris_regions_lst[v[0]].iris_region
            iris_l = self.iris_regions_lst[v[1]].iris_region
            self.nodes[v]['iris'] = iris_k.Intersection(iris_l)

    def optimize_points(self):
        d = self.iris_regions_lst[0].iris_region.ambient_dimension()
        x = cp.Variable((self.number_of_nodes(), d))
        x.value = np.array([self.nodes[v]['iris'].ChebyshevCenter() for v in self.nodes])

        constraints = []
        for i, v in enumerate(self.nodes):
            A_current = self.nodes[v]['iris'].A()
            b_current = self.nodes[v]['iris'].b()
            constraints.append(A_current @ x[i] <= b_current)

        A = nx.incidence_matrix(self, oriented=True)
        y = A.T.dot(x)
        cost = cp.sum(cp.norm(y, 2, axis=1))

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver='CLARABEL')

        for i, v in enumerate(self.nodes):
            self.nodes[v]['point'] = x[i].value

    @staticmethod
    def node(k, l):
        return (k, l) if k < l else (l, k)

    def regionsContainingPoint(self, point: np.array) -> List[int]:
        ir_idx = 0
        ir_idx_lst = []
        # populate list of IRIS regions containing point
        for ir in self.iris_regions_lst:
            if ir.isPointSafe(point):
                ir_idx_lst.append(ir_idx)
            ir_idx += 1

        # check if no regions contains point
        if len(ir_idx_lst) == 0:
            raise ValueError("No IRIS region contains the point.")

        return ir_idx_lst

    def computeShortestPath(self, goal: np.array):
        tic = time()

        rows = []
        data = []
        for k in self.regionsContainingPoint(goal):      # should return list
            for l in self.iris_intersections[k]:
                v = self.node(k, l)
                i = self.v2i[v]
                rows.append(i)
                pv = self.nodes[v]['point']
                data.append(np.linalg.norm(goal - pv))
        cols = [0] * len(rows)
        shape = (len(self.nodes), 1)
        adj_col = sp.sparse.csr_matrix((data, (rows, cols)), shape)
        adj_mat = sp.sparse.bmat([[self.adj_mat, adj_col], [adj_col.T, None]])

        dist, succ = sp.sparse.csgraph.dijkstra(
            csgraph=adj_mat,
            directed=False,
            return_predecessors=True,
            indices=-1
        )

        planner = lambda start: self._planner_all_to_one(start, dist, succ)

        return planner, time() - tic

    def _planner_all_to_one(self, start, dist, succ):
        tic = time()

        length = np.inf
        for k in self.regionsContainingPoint(start):
            for l in self.iris_intersections[k]:
                v = self.node(k, l)
                i = self.v2i[v]
                dist_vg = dist[i]
                if np.isinf(dist_vg):
                    return None, None
                dist_sv = np.linalg.norm(self.nodes[v]['point'] - start)
                dist_sg = dist_sv + dist_vg
                if dist_sg < length:
                    length = dist_sg
                    first_box = k
                    first_vertex = i

        box_sequence = self._succ_to_box_sequence(succ, first_box, first_vertex)

        return box_sequence, length, time() - tic

    def _succ_to_box_sequence(self, succ, first_box, first_vertex):

        box_sequence = [first_box]
        i = first_vertex
        while succ[i] >= 0:
            v = self.i2v[i]
            j = succ[i]
            if succ[j] >= 0:
                w = self.i2v[j]
                while box_sequence[-1] in w:
                    i = j
                    v = w
                    j = succ[i]
                    w = self.i2v[j]
            if v[0] == box_sequence[-1]:
                box_sequence.append(v[1])
            else:
                box_sequence.append(v[0])
            i = succ[i]

        return box_sequence

