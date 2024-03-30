from typing import List

import numpy as np

from pnc.planner.multicontact.iris_seq_planner import IrisGraph
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

    def addIris(self, iris_processor_lst):
        for ir in iris_processor_lst:
            self.iris_list.append(ir)

    def computeIris(self):
        for ir in self.iris_list:
            ir.computeIris()

    def areIrisSeedsContained(self):
        start_pos = self.iris_start_seed
        goal_pos = self.iris_goal_seed

        if self.iris_list[0].isPointSafe(goal_pos) or \
                self.iris_list[-1].isPointSafe(start_pos):
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

        # sample from within staring IRIS region, compute new IRIS region and check
        # if it intersects with ending IRIS region
        # new_seed = self.iris_list[0].iris_region.UniformSample(RandomGenerator(), self.iris_start_seed)
        obstacles = self.iris_list[0].obstacles_mut
        domain = self.iris_list[0].domain_mut

        # sample random seed between start and goal IRIS regions
        start_centroid = self.iris_list[0].iris_region.ChebyshevCenter()
        goal_centroid = self.iris_list[1].iris_region.ChebyshevCenter()
        new_seed = np.random.normal(loc=(start_centroid+goal_centroid)/2, scale=[0.05, 0.1, 0.15])

        # check that new seed is not in collision before creating new IRIS region
        b_resample = self.pointInCollision(new_seed)
        while b_resample:
            new_seed = np.random.normal(loc=(start_centroid + goal_centroid) / 2, scale=[0.05, 0.1, 0.45])
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

    def visualize(self, meshcat_viewer, frame_name):
        for i, ir in enumerate(self.iris_list):
            ir.visualize(meshcat_viewer, frame_name + '/' + str(i))

    def findShortestPath(self, start: np.array,
                         goal: np.array) -> List[int]:
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
            iris_seq = list(iris_p_init)
        else:
            iris_seq, length, runtime = planner(start)
        return iris_seq

    def regionsContainingPoint(self, point: np.array) -> List[int]:
        """
        Find the IRIS region that contains the given point.
        :param point: [np.array] 3D point
        :return: [int] IRIS region index
        """
        return self.iris_graph.regionsContainingPoint(point)

    def getIrisGraph(self):
        return self.iris_graph

    def getIrisRegions(self):
        return self.iris_list
