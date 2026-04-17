from typing import List

import numpy as np
import scipy as sp

from .iris_seq_planner import IrisGraph
from .iris_geom_interface import IrisGeomInterface
from pydrake.common import RandomGenerator


class IrisRegionsManager:
    def __init__(self,
                 iris_start: IrisGeomInterface,
                 iris_goal: IrisGeomInterface = None):
        self.iris_list = []
        self.iris_list.append(iris_start)
        self.iris_start_seed = iris_start.seed_pos

        if iris_goal is not None:
            self.iris_list.append(iris_goal)
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

    def areIrisListSeedsContained(self):
        # check containment
        for ir_num, ir in enumerate(self.iris_list):
            if ir_num < len(self.iris_list) - 1:
                curr_pos = self.iris_list[ir_num].seed_pos
                next_pos = self.iris_list[ir_num+1].seed_pos
                b_goal_pos_in_prev_iris = self.iris_list[ir_num].isPointSafe(next_pos)
                b_goal_pos_in_next_iris = self.iris_list[ir_num+1].isPointSafe(curr_pos)

                # store index of IRIS region(s) containing both start and goal seeds
                if b_goal_pos_in_prev_iris:
                    self.global_iris.append([ir_num])
                if b_goal_pos_in_next_iris:
                    self.global_iris.append([ir_num+1])

        # return whether a global IRIS region exists
        if len(self.global_iris) > 0:
            return True
        else:
            return False

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

    def connectIrisListSeeds(self, choose_iris_by: str = None,
                             hint: np.ndarray = None):
        """
        Connect the list of IRIS regions.
        This approach is based on checking if the IRIS region in the iris_list are connected.
        If they are not connected, a new sample is obtained by sampling in between centroids.
        An alternative approach would be to do an RRT-based expansion.
        :return: [None] Stores the new IRIS regions in the iris_list and graph in iris_graph
        """
        if choose_iris_by is None:
            choose_iris_by = "centroid"

        # If IRIS regions span from start to goal seeds, heuristically initialize graph and iris_seq
        b_single_iris = self.areIrisListSeedsContained()
        if b_single_iris:
            self.iris_graph = None

            # Prioritize IRIS region already containing both start/goal seeds
            if len(self.global_iris) == 1:
                self.iris_idx_seq = [self.global_iris[0]]
            elif choose_iris_by == "volume":
                max_vol, max_idx = 0, 0
                for ir_num, ir in enumerate(self.iris_list):
                    curr_vol = ir.iris_region.MaximumVolumeInscribedEllipsoid().CalcVolume()
                    if curr_vol > max_vol:
                        max_vol = curr_vol
                        max_idx = ir_num
                self.iris_idx_seq = [max_idx]
            else:
                # choose the one w/centroid closest to goal seed (alternative to choosing largest volume)
                c_ellipse_start_iris = self.iris_list[0].iris_region.MaximumVolumeInscribedEllipsoid().center()
                c_ellipse_goal_iris = self.iris_list[1].iris_region.MaximumVolumeInscribedEllipsoid().center()
                c_frame_traj = (self.iris_start_seed + self.iris_goal_seed) / 2.0
                dist_start_trajc = np.linalg.norm(c_ellipse_start_iris - c_frame_traj)
                dist_goal_trajc = np.linalg.norm(c_ellipse_goal_iris - c_frame_traj)
                self.iris_idx_seq = [0] if dist_start_trajc < dist_goal_trajc else [1]
            return

        # sample from within staring IRIS region, compute new IRIS region and check
        # if it intersects with ending IRIS region
        obstacles = self.iris_list[0].obstacles_mut
        domain = self.iris_list[0].domain_mut

        # sample random seed between start and goal IRIS regions
        extended_iris_list = []
        for ir_num, ir in enumerate(self.iris_list):
            if ir_num >= len(self.iris_list) - 1:
                break
            if hint is not None:
                new_seed = hint
            else:
                start_centroid = self.iris_list[ir_num].seed_pos
                goal_centroid = self.iris_list[ir_num+1].seed_pos
                # start_centroid = self.iris_list[ir_num].iris_region.ChebyshevCenter()
                # goal_centroid = self.iris_list[ir_num+1].iris_region.ChebyshevCenter()
                new_seed = np.random.normal(loc=(start_centroid+goal_centroid)/2, scale=[0.05, 0.01, 0.1])
                new_seed[2] = (self.iris_list[ir_num].iris_region.ChebyshevCenter()[2] + self.iris_list[ir_num+1].iris_region.ChebyshevCenter()[2])/2

            # check that new seed is not in collision before creating new IRIS region
            b_resample = self.pointInCollision(new_seed)
            while b_resample:
                new_seed = np.random.normal(loc=(start_centroid + goal_centroid) / 2, scale=[0.05, 0.05, 0.15])
                b_resample = self.pointInCollision(new_seed)

            # create IRIS region using collision-free seed
            new_iris = IrisGeomInterface(obstacles, domain, new_seed)

            # append new IRIS processor and compute IRIS region
            new_iris.computeIris()
            extended_iris_list.append(new_iris)

            # create IRIS regions until the seeds connect
            b_done = False
            while not b_done:
                if (new_iris.iris_region.IntersectsWith(self.iris_list[ir_num].iris_region)
                        and new_iris.iris_region.IntersectsWith(self.iris_list[ir_num+1].iris_region)):
                    b_done = True
                else:
                    # update seed and create new IRIS region
                    new_seed = self.iris_list[-1].iris_region.UniformSample(RandomGenerator(), new_seed)
                    new_iris = IrisGeomInterface(obstacles, domain, new_seed)
                    self.iris_list.append(new_iris)
                    self.iris_list[-1].computeIris()
        self.addIris(extended_iris_list)
        self.iris_graph = IrisGraph(self.iris_list)

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
        start_centroid = np.array([0.35, 0, 0.55])
        goal_centroid = np.array([0.35, 0, 0.55])
        new_seed = np.random.normal(loc=(start_centroid+goal_centroid)/2, scale=[0.001, 0.01, 0.01])

        # check that new seed is not in collision before creating new IRIS region
        b_resample = self.pointInCollision(new_seed)
        while b_resample:
            new_seed = np.random.normal(loc=(start_centroid + goal_centroid) / 2, scale=[0.05, 0.05, 0.15])
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
                         goal: np.array,
                         hint_iris: int = None) -> List[int]:
        """
        First, try to use logic to find the right sequence of IRIS regions.
        If too convoluted, use the graph.
        """

        # If total number of IRIS regions is 1 or 2, we can use logic to find the right sequence
        if len(self.iris_list) == 1 and len(self.global_iris) == 1:
            return [0]
        elif len(self.iris_list) == 2 and len(self.global_iris) == 1:
            return self.global_iris[0]
        elif len(self.iris_list) == 2 and len(self.global_iris) == 2:
            b_last_global_contains_start = self.iris_list[self.global_iris[1][0]].isPointSafe(start)
            b_last_global_contains_goal = self.iris_list[self.global_iris[1][0]].isPointSafe(goal)
            b_first_global_contains_goal = self.iris_list[self.global_iris[0][0]].isPointSafe(goal)
            b_first_global_contains_start = self.iris_list[self.global_iris[0][0]].isPointSafe(start)
            if b_last_global_contains_start and b_last_global_contains_goal:
                # if the start/goal point is contained in 2nd global IRIS, favor it
                return self.global_iris[1]
            elif not b_last_global_contains_goal and b_first_global_contains_goal and b_first_global_contains_start:
                # by construction, these should be true
                return self.global_iris[0]
            elif b_first_global_contains_start and b_first_global_contains_goal:
                # by construction, these should be true
                return self.global_iris[0]
        else:
            if self.iris_graph is None:
                self.iris_graph = IrisGraph(self.iris_list)

            # look at regions that contain the start and goal points
            regions_containing_goal = self.iris_graph.regionsContainingPoint(goal)
            regions_containing_start = self.iris_graph.regionsContainingPoint(start)
            b_overlapping_regions = set(regions_containing_goal).intersection(set(regions_containing_start))

            if (len(regions_containing_goal) >= 2 or len(regions_containing_start) >= 2) and b_overlapping_regions:
                # check if there is a path between any two of these regions
                max_pair = None
                max_vol = -np.inf
                for idx1 in range(len(regions_containing_start)):
                    for idx2 in range(len(regions_containing_goal)):
                        i1 = regions_containing_start[idx1]
                        i2 = regions_containing_goal[idx2]
                        vol = self.iris_list[i1].iris_region.Intersection(
                            self.iris_list[i2].iris_region
                        ).MaximumVolumeInscribedEllipsoid().CalcVolume()
                        if vol > max_vol:
                            max_vol = vol
                            max_pair = (i1, i2)
                if max_pair is not None:
                    if max_pair[0] == max_pair[1]:
                        # if both indices are the same, return that IRIS region
                        return [max_pair[0]]
                    else:
                        return list(max_pair)

            # Reaching here means we cannot reasonably go from start to goal with a single IRIS region pair.
            # Resorting to Dijkstra's algorithm to find the shortest path using IRIS centroids
            print(f"Creating Graph for (start, goal): ({start}, {goal})")
            self.iris_graph = IrisGraph(self.iris_list)

        # compute shortest path from start to goal
        planner, runtime = self.iris_graph.computeShortestPath(goal)
        iris_seq_tmp, _, __ = planner(start)

        # If <= 2 IRIS regions in shortest path, check that the start/goal are not contained in one IRIS
        if len(iris_seq_tmp) <= 2:
            if hint_iris is not None:
                start_is_in_hint = self.iris_list[hint_iris].isPointSafe(start)
                goal_is_in_hint = self.iris_list[hint_iris].isPointSafe(goal)
                if hint_iris in iris_seq_tmp and start_is_in_hint and goal_is_in_hint:
                    return [hint_iris]
                else:
                    return iris_seq_tmp
            else:
                # choose the one whose centroid is closest to the goal seed
                centroid_dist = []
                for ir in iris_seq_tmp:
                    centroid_dist.append(sp.linalg.norm(self.iris_list[ir].iris_region.ChebyshevCenter() - goal))
                return [iris_seq_tmp[np.argmin(centroid_dist)]]

        # If we reach this point, there are > 2 IRIS regions in shortest path
        return iris_seq_tmp

    def findShortestPathOld(self, start: np.array,
                         goal: np.array,
                         hint_iris: int = None) -> List[int]:
        # if single IRIS region, return the index of corresponding global IRIS region
        if self.iris_graph is None:
            if len(self.iris_list) == 1 and len(self.global_iris) == 1:
                return [self.iris_idx_seq[0]]
            elif len(self.iris_list) == 2 and len(self.global_iris) == 1:
                return self.global_iris[0]
            elif len(self.iris_list) == 2 and len(self.global_iris) == 2:
                b_last_global_contains_start = self.iris_list[self.global_iris[1][0]].isPointSafe(start)
                b_last_global_contains_goal = self.iris_list[self.global_iris[1][0]].isPointSafe(goal)
                b_first_global_contains_goal = self.iris_list[self.global_iris[0][0]].isPointSafe(goal)
                b_first_global_contains_start = self.iris_list[self.global_iris[0][0]].isPointSafe(start)
                if b_last_global_contains_start and b_last_global_contains_goal:
                    # if the start/goal point is contained in 2nd global IRIS, favor it
                    return self.global_iris[1]
                elif not b_last_global_contains_goal and b_first_global_contains_goal and b_first_global_contains_start:
                    # by construction, these should be true
                    return self.global_iris[0]
                elif b_first_global_contains_start and b_first_global_contains_goal:
                    # by construction, these should be true
                    return self.global_iris[0]
            else:
                # for some reason we didn't need the graph before
                print(f"Creating Graph for seed: {self.iris_start_seed}")
                self.iris_graph = IrisGraph(self.iris_list)

        planner, runtime = self.iris_graph.computeShortestPath(goal)

        # find which IRIS region contains the start point (first check hint_iris_region)
        if hint_iris is not None:
            if self.iris_list[hint_iris].isPointSafe(start):
                iris_p_init = hint_iris
            else:
                iris_p_init = self.iris_graph.regionsContainingPoint(start)[0]
        else:
            ir_start_idx = 0
            for ir in self.iris_list:
                if ir.isPointSafe(start):
                    iris_p_init = ir_start_idx
                    break
                ir_start_idx += 1

        # find first IRIS region contains the goal point
        ir_goal_idx = 0
        for ir in self.iris_list:
            if ir.isPointSafe(goal):
                iris_p_goal = ir_goal_idx
                break
            ir_goal_idx += 1

        regions_containing_goal = self.iris_graph.regionsContainingPoint(goal)
        # if several regions contain the goal, check which one overlaps the most
        if len(regions_containing_goal) > 1:
            intersect_vol = []
            init_iris_region = self.iris_list[iris_p_init].iris_region
            for r in regions_containing_goal:
                test_iris_region = self.iris_list[r].iris_region
                intersect_vol.append(init_iris_region.Intersection(test_iris_region).MaximumVolumeInscribedEllipsoid().CalcVolume())
            iris_p_goal = regions_containing_goal[np.argmax(intersect_vol)]
            return [iris_p_goal]

        iris_seq_tmp, length, runtime = planner(start)
        if iris_p_init == iris_p_goal:
            iris_seq = [iris_p_init]
        elif len(regions_containing_goal) == 1 and (len(iris_seq_tmp) == 2):
            iris_seq = [iris_p_init, iris_p_goal]
        elif len(regions_containing_goal) >= 1 and (len(iris_seq_tmp) >= 2):
            # if goal is contained in > 1 IRIS region, check if init and goal IRIS regions intersect
            if (self.iris_list[iris_p_init].irisIntersects(self.iris_list[iris_p_goal].iris_region)):
                if len(iris_seq_tmp) == 2:
                    iris_seq = [iris_p_init, iris_p_goal]
                elif len(iris_seq_tmp) == 3:
                    # iris_seq = [iris_p_init, iris_p_init, iris_p_goal]
                    iris_seq = iris_seq_tmp
                else:
                    raise NotImplementedError
            else:
                print(f"[Iris Region Manager] Check IRIS sequence from {start} to {goal}")
                iris_seq = iris_seq_tmp
        else:   # need to traverse more than 2 IRIS regions?
            print(f"[Iris Region Manager] Check if goal is contained in IRIS sequence from {start} to {goal}.")
            iris_seq = iris_seq_tmp
        return iris_seq

    def regionsContainingPoint(self, point: np.array) -> List[int]:
        """
        Find the IRIS region that contains the given point.
        :param point: [np.array] 3D point
        :return: [int] IRIS region index
        """
        # We reserve None for single IRIS regions containing from start to goal seeds
        if self.iris_graph is None:
            # if we already agreed we can stay in the initial IRIS region, proceed with that
            if len(self.iris_idx_seq) == 1:
                # there are some type inconsistency issues. Fix later
                return [self.iris_idx_seq[0]] if isinstance(self.iris_idx_seq[0], int) else self.iris_idx_seq[0]
            # point must be contained in either the start/goal IRIS region
            elif len(self.global_iris) == 1 and self.iris_list[self.iris_idx_seq[0]].isPointSafe(point):
                return [self.iris_idx_seq[0]]
            # if self.iris_list[self.iris_idx_seq].isPointSafe(point):
            #     return [self.iris_idx_seq]
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
