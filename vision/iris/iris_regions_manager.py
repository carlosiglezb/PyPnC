import numpy as np
from iris_geom_interface import IrisGeomInterface
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

    def connectIrisSeeds(self):
        """
        Connect start and goal IRIS seeds.
        The current approach is based on sampling and growing from the starting IRIS region.
        An alternative approach would be to do an RRT-based expansion.
        :return: [None] Stores the new IRIS regions in the iris_list
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

    def visualize(self, meshcat_viewer):
        for i, ir in enumerate(self.iris_list):
            ir.visualize(meshcat_viewer, i)
