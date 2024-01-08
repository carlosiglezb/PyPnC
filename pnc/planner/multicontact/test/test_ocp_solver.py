import unittest

import numpy as np

from collections import OrderedDict
from pnc.planner.multicontact.fastpathplanning.smooth import optimize_multiple_bezier
from pnc.planner.multicontact.frame_traversable_region import FrameTraversableRegion
from pnc.planner.multicontact.locomanipulation_frame_planner import LocomanipulationFramePlanner

class TestOCPSolver(unittest.TestCase):

    def get_sample_collision_free_boxes(self):
        # save safe box regions
        box_llim, box_ulim = OrderedDict(), OrderedDict()

        # lower bounds of end-effectors safe boxes
        box_llim['LF'] = np.array([
            [-0.2, 0.0, 0.0],  # prevent leg-crossing
            [-0.1, 0.0, 0.4],  # prevent leg-crossing
            [0.4, 0.0, 0.0]  # prevent leg-crossing
        ])
        box_llim['LH'] = np.array([
            [-0.1, 0.0, 0.7],  # prevent leg-crossing
            [0.1, 0.0, 0.7],  # prevent leg-crossing
            [0.5, 0.0, 0.7]  # prevent leg-crossing
        ])

        # upper bounds of the safe boxes
        box_ulim['LF'] = np.array([
            [0.25, 0.35, 0.6],  # z stops at kin. limit
            [0.6, 0.4, 0.6],  # x stops at kin. limit
            [0.8, 0.4, 0.6]  # x stops at kin. limit
        ])
        box_ulim['LH'] = np.array([
            [0.15, 0.45, 1.3],  # prevent leg-crossing
            [0.55, 0.38, 1.3],  # prevent leg-crossing
            [0.8, 0.45, 1.3]  # prevent leg-crossing
        ])
        return box_llim, box_ulim

    def test_optim_bezier_lf_arch_motion(self):

        reach_region = [1]  # override
        L, U, durations, safe_points_lst, fixed_frames = [], [], [], [], []

        # collision-free boxes
        box_llim, box_ulim = self.get_sample_collision_free_boxes()

        # manually assign lower limits
        L.append({'LF': np.vstack([[box_llim['LF'][0]] * 2])})
        L.append({'LF': np.vstack([box_llim['LF'][0], box_llim['LF'][1], box_llim['LF'][2]])})
        L.append({'LF': np.vstack([[box_llim['LF'][2]]])})

        # manually assign upper limits
        U.append({'LF': np.vstack([box_ulim['LF'][0] * 2])})
        U.append({'LF': np.vstack([box_ulim['LF'][0], box_ulim['LF'][1], box_ulim['LF'][2]])})
        U.append({'LF': np.vstack([box_ulim['LF'][2]])})

        durations.append({'LF': np.array([[0.3], [0.3]])})
        durations.append({'LF': np.array([0.2] * 3)})
        durations.append({'LF': np.array([[0.3], [0.3]])})
        alpha = {1: 1, 2: 2, 3: 3}

        # Safe points for initial and last positions
        safe_points_lst.append({'LF': np.array([0.06, 0.14, 0.])})
        safe_points_lst.append({'LF': np.array([0.06, 0.14, 0.])})
        safe_points_lst.append({'LF': np.array([0.5, 0.14, 0.])})

        fixed_frames.append({'LF'})
        fixed_frames.append({})

        path, sol_stats = optimize_multiple_bezier(reach_region, None, L, U, durations, alpha, safe_points_lst, fixed_frames)
        self.assertTrue(sol_stats['cost'] < 1e9, "Problem seems to be infeasible")

        # Visualizer
        traversable_region = FrameTraversableRegion('LF', b_visualize_safe=True)
        traversable_region.load_collision_free_boxes(box_llim['LF'], box_ulim['LF'])
        visualizer = traversable_region.get_visualizer()

        # Plot solution
        i = 0
        for p in path:
            for seg in range(len(p.beziers)):
                bezier_curve = [p.beziers[seg]]
                LocomanipulationFramePlanner.visualize_bezier_points(visualizer, 'LF', bezier_curve, seg)
            i += 1
        self.assertTrue(True, "Check solution in visualizer")

    def test_optim_bezier_lfh_arch_motion(self):

        reach_region = [1, 1]  # override
        L, U, durations, safe_points_lst, fixed_frames = [], [], [], [], []

        # collision-free boxes
        box_llim, box_ulim = self.get_sample_collision_free_boxes()

        # manually assign lower limits
        L.append({'LF': np.vstack([[box_llim['LF'][0]] * 2]),
                  'LH': np.vstack([box_llim['LH'][1], box_llim['LH'][1]])})
        L.append({'LF': np.vstack([box_llim['LF'][0], box_llim['LF'][1], box_llim['LF'][2]]),
                  'LH': np.vstack([[box_llim['LH'][1]] * 3])})
        L.append({'LF': np.vstack([box_llim['LF'][2]]),
                  'LH': np.vstack([box_llim['LH'][1]])})

        # manually assign upper limits
        U.append({'LF': np.vstack([box_ulim['LF'][0] * 2]),
                  'LH': np.vstack([box_ulim['LH'][1], box_llim['LH'][1]])})
        U.append({'LF': np.vstack([box_ulim['LF'][0], box_ulim['LF'][1], box_ulim['LF'][2]]),
                  'LH': np.vstack([[box_ulim['LH'][1]] * 3])})
        U.append({'LF': np.vstack([box_ulim['LF'][2]]),
                  'LH': np.vstack([box_ulim['LH'][1]])})

        durations.append({'LF': np.array([[0.3], [0.3]]),
                          'LH': np.array([[0.3], [0.3]])})
        durations.append({'LF': np.array([0.2] * 3),
                          'LH': np.array([0.2] * 3)})
        durations.append({'LF': np.array([[0.3], [0.3]]),
                          'LH': np.array([[0.3], [0.3]])})
        alpha = {1: 1, 2: 2, 3: 3}

        # Safe points for initial and last positions
        safe_points_lst.append({'LF': np.array([0.06, 0.14, 0.]),
                                'LH': np.array([0.22, 0.3, 0.74])})
        safe_points_lst.append({'LF': np.array([0.06, 0.14, 0.]),
                                'LH': np.array([0.3, 0.37, 0.89])})
        safe_points_lst.append({'LF': np.array([0.5, 0.14, 0.]),
                                'LH': np.array([0.3, 0.37, 0.89])})

        fixed_frames.append({'LF'})
        fixed_frames.append({'LH'})

        path, sol_stats = optimize_multiple_bezier(reach_region, None, L, U, durations, alpha, safe_points_lst, fixed_frames)
        self.assertTrue(sol_stats['cost'] < 1e9, "Problem seems to be infeasible")

        # Visualizer
        traversable_region = OrderedDict()
        traversable_region['LF'] = FrameTraversableRegion('LF', b_visualize_safe=True)
        traversable_region['LF'].load_collision_free_boxes(box_llim['LF'], box_ulim['LF'])
        visualizer = traversable_region['LF'].get_visualizer()
        traversable_region['LH'] = FrameTraversableRegion('LH', b_visualize_safe=True,
                                              visualizer=visualizer)
        traversable_region['LH'].load_collision_free_boxes(box_llim['LH'], box_ulim['LH'])


        # Plot solution
        i = 0
        for p in path:
            for seg in range(len(p.beziers)):
                bezier_curve = [p.beziers[seg]]
                if i == 0:
                    LocomanipulationFramePlanner.visualize_bezier_points(visualizer, 'LF', bezier_curve, seg)
                elif i == 1:
                    LocomanipulationFramePlanner.visualize_bezier_points(visualizer, 'LH', bezier_curve, seg)
            i += 1
        self.assertTrue(True, "Check solution in visualizer")


if __name__ == '__main__':
    unittest.main()
