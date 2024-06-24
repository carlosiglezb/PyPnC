from typing import List

import numpy as np

from pnc.planner.multicontact.kin_feasibility.iris_seq_planner import IrisGraph
from vision.iris.iris_geom_interface import IrisGeomInterface
from pydrake.common import RandomGenerator


class IrisRegionsManager:
    def __init__(self,
                 iris_start: IrisGeomInterface,
                 iris_goal: IrisGeomInterface):
        self.iris_list = []
        self.iris_list.append(iris_start)
        self.iris_list.append(iris_goal)

        self.iris_start_seed = iris_start.seed_pos
        self.iris_goal_seed = iris_goal.seed_pos
        self.iris_graph = None      # this is computed after connecting IRIS seeds
        self.iris_idx_seq = []    # this is computed after finding the shortest path
        self.global_iris = []     # index of IRIS region containing start & goal seeds

    def addIris(self, iris_processor_lst):
        for ir in iris_processor_lst:
            self.iris_list.append(ir)

    def computeIris(self):
        for ir in self.iris_list:
            ir.computeIris()

    def areIrisSeedsContained(self):
        start_pos = self.iris_start_seed
        goal_pos = self.iris_goal_seed

        # check containment
        b_goal_pos_in_start_iris = self.iris_list[0].isPointSafe(goal_pos)
        b_start_pos_in_goal_iris = self.iris_list[1].isPointSafe(start_pos)

        # store index of IRIS region(s) containing both start and goal seeds
        if b_goal_pos_in_start_iris:
            self.global_iris.append([0])
        if b_start_pos_in_goal_iris:
            self.global_iris.append([1])

        # return whether a global IRIS region exists
        if b_start_pos_in_goal_iris or b_goal_pos_in_start_iris:
            return True
        else:
            return False

    def pointInCollision(self, point: np.array):
        """
        Assuming the obstacles are the same for all IRIS regions, we check that
        the point is not contained in any of the obstacles provided in the first
        IRIS region list.
        """
        for obs in self.iris_list[0].obstacles_mut:
            if obs.PointInSet(point):
                return True
        return False

    def connectIrisSeeds(self):
        """
        Connect start and goal IRIS seeds.
        The current approach is based on sampling and growing from the starting IRIS region.
        An alternative approach would be to do an RRT-based expansion.
        :return: [None] Stores the new IRIS regions in the iris_list and graph in iris_graph
        """

        # If IRIS regions span from start to goal seeds, heuristically initialize graph and iris_seq
        b_single_iris = self.areIrisSeedsContained()
        if b_single_iris:
            self.iris_graph = None

            # Prioritize IRIS region already containing both start/goal seeds
            if len(self.global_iris) == 1:
                self.iris_idx_seq = self.global_iris[0]
            else:
                # choose the one w/centroid closes to goal seed (alternative to choosing largest volume)
                c_ellipse_start_iris = self.iris_list[0].iris_region.MaximumVolumeInscribedEllipsoid().center()
                c_ellipse_goal_iris = self.iris_list[1].iris_region.MaximumVolumeInscribedEllipsoid().center()
                c_frame_traj = (self.iris_start_seed + self.iris_goal_seed) / 2.0
                dist_start_trajc = np.linalg.norm(c_ellipse_start_iris - c_frame_traj)
                dist_goal_trajc = np.linalg.norm(c_ellipse_goal_iris - c_frame_traj)
                self.iris_idx_seq = [0] if dist_start_trajc < dist_goal_trajc else [1]
            return

        # sample from within staring IRIS region, compute new IRIS region and check
        # if it intersects with ending IRIS region
        # new_seed = self.iris_list[0].iris_region.UniformSample(RandomGenerator(), self.iris_start_seed)
        obstacles = self.iris_list[0].obstacles_mut
        domain = self.iris_list[0].domain_mut

        # sample random seed between start and goal IRIS regions
        # start_centroid = self.iris_list[0].iris_region.ChebyshevCenter()
        # goal_centroid = self.iris_list[1].iris_region.ChebyshevCenter()
        # ----------- settings for G1
        # start_centroid = self.iris_start_seed + np.array([0., 0., 0.6])
        # goal_centroid = self.iris_goal_seed + np.array([-0.15, 0., 0.6])
        # ----------- settings for Val
        # start_centroid = self.iris_start_seed + np.array([0., 0., 0.5])
        # goal_centroid = self.iris_goal_seed + np.array([0.0, 0., 0.5])      # <-- ideal for G1
        # goal_centroid = self.iris_goal_seed + np.array([0.15, 0., 0.5])     # <-- ideal for Val
        # ----------- settings for ergoCub
        # start_centroid = self.iris_start_seed + np.array([0., 0., 0.35])
        # goal_centroid = self.iris_goal_seed + np.array([0.0, 0., 0.35])
        # ----------- settings for all
        start_centroid = np.array([0.3, 0, 0.8])
        goal_centroid = np.array([0.3, 0, 0.8])
        new_seed = np.random.normal(loc=(start_centroid+goal_centroid)/2, scale=[0.001, 0.1, 0.1])

        # check that new seed is not in collision before creating new IRIS region
        b_resample = self.pointInCollision(new_seed)
        while b_resample:
            new_seed = np.random.normal(loc=(start_centroid + goal_centroid) / 2, scale=[0.05, 0.05, 0.25])
            b_resample = self.pointInCollision(new_seed)

        # create IRIS region using collision-free seed
        new_iris = IrisGeomInterface(obstacles, domain, new_seed)

        # append new IRIS processor and compute IRIS region
        new_iris.computeIris()
        self.iris_list.append(new_iris)

        # get goal IRIS region
        goal_IRIS = self.iris_list[1].iris_region

        # create IRIS regions until the seeds connect
        b_done = False
        while not b_done:
            if new_iris.iris_region.IntersectsWith(goal_IRIS):
                b_done = True
            else:
                # update seed and create new IRIS region
                new_seed = self.iris_list[-1].iris_region.UniformSample(RandomGenerator(), new_seed)
                new_iris = IrisGeomInterface(obstacles, domain, new_seed)
                self.iris_list.append(new_iris)
                self.iris_list[-1].computeIris()
        self.iris_graph = IrisGraph(self.iris_list)

    def visualize(self, meshcat_viewer, frame_name='frame'):
        for i, ir in enumerate(self.iris_list):
            ir.visualize(meshcat_viewer, frame_name + '/' + str(i))

    def findShortestPath(self, start: np.array,
                         goal: np.array) -> List[int]:
        # if single IRIS region, return the index of corresponding global IRIS region
        if self.iris_graph is None:
            return [self.iris_idx_seq[0]]

        planner, runtime = self.iris_graph.computeShortestPath(goal)

        # find which IRIS region contains the start point
        ir_start_idx = 0
        for ir in self.iris_list:
            if ir.isPointSafe(start):
                iris_p_init = ir_start_idx
                break
            ir_start_idx += 1

        # find which IRIS region contains the goal point
        ir_goal_idx = 0
        for ir in self.iris_list:
            if ir.isPointSafe(goal):
                iris_p_goal = ir_goal_idx
                break
            ir_goal_idx += 1

        if iris_p_init == iris_p_goal:
            iris_seq = [iris_p_init]
        else:
            iris_seq, length, runtime = planner(start)
        return iris_seq

    def regionsContainingPoint(self, point: np.array) -> List[int]:
        """
        Find the IRIS region that contains the given point.
        :param point: [np.array] 3D point
        :return: [int] IRIS region index
        """
        # We reserve None for single IRIS regions containing from start to goal seeds
        if self.iris_graph is None:
            # point must be contained in either the start/goal IRIS region
            if self.iris_list[self.iris_idx_seq[0]].isPointSafe(point):
                return [self.iris_idx_seq[0]]
            elif len(self.global_iris) > 1:
                # as fallback, loop through IRIS regions in the global IRIS list
                for gi in self.global_iris:
                    if gi[0] != self.iris_idx_seq[0]:
                        print(f"[IRIS fallback] Using IRIS region {gi[0]} for point {point}.")
                        return [gi[0]]
            else:
                raise ValueError(f"Ideal IRIS Region does NOT contain point {point}.")
        return self.iris_graph.regionsContainingPoint(point)

    def getIrisGraph(self):
        return self.iris_graph

    def getIrisRegions(self):
        return self.iris_list
