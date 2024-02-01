import unittest

import os
import sys
import numpy as np
from ruamel.yaml import YAML

import pnc.planner.multicontact.fastpathplanning.fastpathplanning as fpp

from collections import OrderedDict
from pnc.planner.multicontact.fastpathplanning.smooth import optimize_multiple_bezier
from pnc.planner.multicontact.frame_traversable_region import FrameTraversableRegion
from pnc.planner.multicontact.locomanipulation_frame_planner import LocomanipulationFramePlanner
from pnc.planner.multicontact.fastpathplanning.polygonal import solve_min_reach_distance
from pnc.planner.multicontact.planner_surface_contact import PlannerSurfaceContact
from util import util

cwd = os.getcwd()
sys.path.append(cwd)

blue = [0., 0., 1., 0.8]


class TestOCPSolver(unittest.TestCase):

    def get_sample_collision_free_boxes(self):
        # save safe box regions
        box_llim, box_ulim = OrderedDict(), OrderedDict()

        # lower bounds of end-effectors safe boxes
        box_llim['LF'] = np.array([
            [-0.2, 0.0, 0.0],  # prevent leg-crossing
            [-0.2, 0.0, 0.4],  # prevent leg-crossing
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
            [0.8, 0.4, 0.6],  # x stops at kin. limit
            [0.8, 0.4, 0.6]  # x stops at kin. limit
        ])
        box_ulim['LH'] = np.array([
            [0.15, 0.45, 1.3],  # prevent leg-crossing
            [0.55, 0.38, 1.3],  # prevent leg-crossing
            [0.8, 0.45, 1.3]  # prevent leg-crossing
        ])
        return box_llim, box_ulim

    def test_optim_bezier_lf_arch_motion(self):

        reach_region = None  # override
        L, U, durations, safe_points_lst, fixed_frames = [], [], [], [], []

        # collision-free boxes
        box_llim, box_ulim = self.get_sample_collision_free_boxes()

        # manually assign lower limits
        L.append({'LF': np.vstack([[box_llim['LF'][0]]])})
        L.append({'LF': np.vstack([box_llim['LF'][0], box_llim['LF'][1], box_llim['LF'][2]])})

        # manually assign upper limits
        U.append({'LF': np.vstack([box_ulim['LF'][0]])})
        U.append({'LF': np.vstack([box_ulim['LF'][0], box_ulim['LF'][1], box_ulim['LF'][2]])})

        durations.append({'LF': np.array([[0.6]])})
        durations.append({'LF': np.array([0.2] * 3)})
        alpha = {1: 1, 2: 2, 3: 3}

        # Safe points for initial and last positions
        safe_points_lst.append({'LF': np.array([0.06, 0.14, 0.])})
        safe_points_lst.append({'LF': np.array([0.06, 0.14, 0.])})
        safe_points_lst.append({'LF': np.array([0.5, 0.14, 0.])})

        # Velocity / acceleration vector (in W frame) of desired end effector position
        lf_contact_front = PlannerSurfaceContact('LF', np.array([0., 0., 1.]))
        surface_normals_lst = [None, lf_contact_front]

        fixed_frames.append({'LF'})
        fixed_frames.append({})

        path, sol_stats = optimize_multiple_bezier(reach_region, None, L, U, durations,
                                                   alpha, safe_points_lst, fixed_frames,
                                                   surface_normals_lst)
        self.assertTrue(sol_stats['cost'] < 1e6, "Problem seems to be infeasible")

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

    def test_optim_bezier_lfk_arch_motion(self):

        reach_region = None
        L, U, durations, safe_points_lst, fixed_frames = [], [], [], [], []

        # collision-free boxes
        box_llim, box_ulim = self.get_sample_collision_free_boxes()

        # manually assign lower limits
        L.append({'LF': np.vstack([[box_llim['LF'][0]]])})
        L.append({'LF': np.vstack([box_llim['LF'][0], box_llim['LF'][1], box_llim['LF'][2]])})

        # manually assign upper limits
        U.append({'LF': np.vstack([box_ulim['LF'][0]])})
        U.append({'LF': np.vstack([box_ulim['LF'][0], box_ulim['LF'][1], box_ulim['LF'][2]])})

        durations.append({'LF': np.array([[0.6]])})
        durations.append({'LF': np.array([0.2] * 3)})
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

        reach_region = None  # override
        L, U, durations, safe_points_lst, fixed_frames = [], [], [], [], []

        # collision-free boxes
        box_llim, box_ulim = self.get_sample_collision_free_boxes()

        # manually assign lower limits
        L.append({'LF': np.vstack([box_llim['LF'][0]]),
                  'LH': np.vstack([box_llim['LH'][1]])})
        L.append({'LF': np.vstack([box_llim['LF'][0], box_llim['LF'][1], box_llim['LF'][2]]),
                  'LH': np.vstack([[box_llim['LH'][1]] * 3])})

        # manually assign upper limits
        U.append({'LF': np.vstack([box_ulim['LF'][0]]),
                  'LH': np.vstack([box_ulim['LH'][1]])})
        U.append({'LF': np.vstack([box_ulim['LF'][0], box_ulim['LF'][1], box_ulim['LF'][2]]),
                  'LH': np.vstack([[box_ulim['LH'][1]] * 3])})

        durations.append({'LF': np.array([[0.6]]),
                          'LH': np.array([[0.6]])})
        durations.append({'LF': np.array([0.2] * 3),
                          'LH': np.array([0.2] * 3)})
        alpha = {1: 1, 2: 2, 3: 3}

        # Safe points for initial and last positions
        safe_points_lst.append({'LF': np.array([0.06, 0.14, 0.]),
                                'LH': np.array([0.22, 0.3, 0.74])})     # initial stance
        safe_points_lst.append({'LF': np.array([0.06, 0.14, 0.]),
                                'LH': np.array([0.3, 0.37, 0.89])})     # end of interval 1
        safe_points_lst.append({'LF': np.array([0.5, 0.14, 0.]),
                                'LH': np.array([0.3, 0.37, 0.89])})     # end of interval 2

        # Velocity / acceleration vector (in W frame) of desired end effector position
        lh_contact_front = PlannerSurfaceContact('LH', np.array([-1, 0., 0.]))
        lf_contact_front = PlannerSurfaceContact('LF', np.array([0, 0., 1.]))
        surface_normals_lst = [lh_contact_front, lf_contact_front]

        # fixed frames
        fixed_frames.append({'LF'})
        fixed_frames.append({'LH'})

        path, sol_stats = optimize_multiple_bezier(reach_region, None, L, U, durations,
                                                   alpha, safe_points_lst, fixed_frames,
                                                   surface_normals_lst)
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

    def test_optim_bezier_lfh_arch_hands_motion(self):

        reach_region = None  # override (two frames)
        L, U, durations, safe_points_lst, fixed_frames = [], [], [], [], []

        # collision-free boxes
        box_llim, box_ulim = self.get_sample_collision_free_boxes()

        # manually assign lower limits
        L.append({'LF': np.vstack([[box_llim['LF'][0]]]),
                  'LH': np.vstack([[box_llim['LH'][1]]])})
        L.append({'LF': np.vstack([box_llim['LF'][0], box_llim['LF'][1], box_llim['LF'][2]]),
                  'LH': np.vstack([[box_llim['LH'][1]] * 3])})
        L.append({'LF': np.vstack([box_llim['LF'][2]]),
                  'LH': np.vstack([box_llim['LH'][1]])})

        # manually assign upper limits
        U.append({'LF': np.vstack([[box_ulim['LF'][0]]]),
                  'LH': np.vstack([[box_ulim['LH'][1]]])})
        U.append({'LF': np.vstack([box_ulim['LF'][0], box_ulim['LF'][1], box_ulim['LF'][2]]),
                  'LH': np.vstack([[box_ulim['LH'][1]] * 3])})
        U.append({'LF': np.vstack([box_ulim['LF'][2]]),
                  'LH': np.vstack([box_ulim['LH'][1]])})

        durations.append({'LF': np.array([[0.6]]),
                          'LH': np.array([[0.6]])})
        durations.append({'LF': np.array([0.2] * 3),
                          'LH': np.array([0.2] * 3)})
        durations.append({'LF': np.array([[0.6]]),
                          'LH': np.array([[0.6]])})
        alpha = {1: 1, 2: 2, 3: 3}

        # Safe points for initial and last positions
        safe_points_lst.append({'LF': np.array([0.06, 0.14, 0.]),
                                'LH': np.array([0.22, 0.3, 0.74])})
        safe_points_lst.append({'LF': np.array([0.06, 0.14, 0.]),
                                'LH': np.array([0.3, 0.37, 0.89])})
        safe_points_lst.append({'LF': np.array([0.5, 0.14, 0.]),
                                'LH': np.array([0.3, 0.37, 0.89])})
        safe_points_lst.append({'LF': np.array([0.5, 0.14, 0.]),
                                'LH': np.array([0.31, 0.36, 0.92])})

        # contact surfaces used for plan
        lh_contact_front = PlannerSurfaceContact('LH', np.array([-1, 0., 0.]))
        lf_contact_front = PlannerSurfaceContact('LF', np.array([0., 0., 1]))
        lh_contact_inner = PlannerSurfaceContact('LH', np.array([0, -1, 0.]))
        lh_contact_inner.set_contact_breaking_velocity(np.array([-1, 0., 0.]))
        surface_normals_lst = [lh_contact_front, lf_contact_front, lh_contact_inner]

        fixed_frames.append({'LF'})
        fixed_frames.append({'LH'})
        fixed_frames.append({'LF'})

        path, sol_stats = optimize_multiple_bezier(reach_region, None, L, U, durations,
                                                   alpha, safe_points_lst, fixed_frames,
                                                   surface_normals_lst)
        self.assertTrue(sol_stats['cost'] < 1e15, "Problem seems to be infeasible")

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

    def test_solve_min_distance_safe(self):
        points_dim = 3      # 3D points
        reach = None
        safe_points_lst = []

        # collision-free boxes
        box_llim, box_ulim = self.get_sample_collision_free_boxes()

        # Safe points being initial and last positions
        safe_points_lst.append({'LF': np.array([0.06, 0.14, 0.])})
        safe_points_lst.append({'LF': np.array([0.5, 0.14, 0.])})

        # Safe boxes
        safe_boxes = OrderedDict()
        safe_boxes['LF'] = fpp.SafeSet(box_llim['LF'], box_ulim['LF'], False)

        # Box sequence
        box_seq = [{'LF': [0, 1, 2]}]     # box sequence for LF

        # Solve minimum distance problem
        traj, length, _ = solve_min_reach_distance(reach, safe_boxes, box_seq, safe_points_lst, None)
        n_points = int(len(traj) / points_dim)
        traj = np.reshape(traj, [n_points, points_dim])

        # visualize
        traversable_region = OrderedDict()
        traversable_region['LF'] = FrameTraversableRegion('LF', b_visualize_safe=True)
        traversable_region['LF'].load_collision_free_boxes(box_llim['LF'], box_ulim['LF'])
        visualizer = traversable_region['LF'].get_visualizer()
        LocomanipulationFramePlanner.visualize_simple_points(visualizer, 'LF', traj)

        naive_distance = 0.4 + 0.44 + 0.4   # simply moving straight up, forward, and down
        self.assertTrue(length < naive_distance, "Trajectory is longer than expected")

        shortest_distance = (np.linalg.norm([0.19, 0.4]) +      # go to [0.25, 0.14, 0.4]
                             0.15 +                             # go to [0.4, 0.14, 0.4]
                             np.linalg.norm([0.1, 0.4]) )       # go to [0.5, 0.14, 0.0]
        self.assertTrue(length < shortest_distance, "Trajectory is longer than expected")

    def test_solve_min_rigid_distance_safe(self):

        aux_frames_path = cwd + '/pnc/reachability_map/output/draco3_aux_frames.yaml'
        aux_frames = LocomanipulationFramePlanner.add_fixed_distance_between_points(aux_frames_path)

        points_dim = 3      # 3D points
        reach = None

        # collision-free boxes
        box_llim, box_ulim = self.get_sample_collision_free_boxes()

        # initial / final foot positions
        lf_init = np.array([0.06, 0.14, 0.])
        lf_final = np.array([0.5, 0.14, 0.])

        # distance between knee and foot frame
        R = util.euler_to_rot([0., np.pi / 6, 0.])
        lknee_lf_offset = R @ np.array([0., -0.00599, 0.324231])  # rotate by 30deg about y

        # Safe points being initial and last positions
        safe_points_lst = []
        safe_points_lst.append({'LF': lf_init,
                                'L_knee': lf_init + lknee_lf_offset})
        safe_points_lst.append({'LF': lf_final,
                                'L_knee': lf_final + lknee_lf_offset})

        # Safe boxes
        safe_boxes = OrderedDict()
        safe_boxes['LF'] = fpp.SafeSet(box_llim['LF'], box_ulim['LF'], False)
        safe_boxes['L_knee'] = fpp.SafeSet(box_llim['LF'], box_ulim['LF'], False)

        # Box sequence
        box_seq = [{'LF': [0, 1, 2], 'L_knee': [0, 1, 2]}]     # box sequence for LF, LK

        # Solve minimum distance problem
        traj, length, _ = solve_min_reach_distance(reach, safe_boxes, box_seq, safe_points_lst, aux_frames)
        n_points = int(len(traj) / points_dim)
        traj = np.reshape(traj, [n_points, points_dim])

        # visualize safe regions
        traversable_region = OrderedDict()
        traversable_region['LF'] = FrameTraversableRegion('LF', b_visualize_safe=True)
        traversable_region['LF'].load_collision_free_boxes(box_llim['LF'], box_ulim['LF'])
        visualizer = traversable_region['LF'].get_visualizer()
        traversable_region['LK'] = FrameTraversableRegion('LK', b_visualize_safe=True, visualizer=visualizer)
        traversable_region['LK'].load_collision_free_boxes(box_llim['LF'], box_ulim['LF'])

        # visualize points
        LocomanipulationFramePlanner.visualize_simple_points(visualizer, 'LF', traj[0:4])
        LocomanipulationFramePlanner.visualize_simple_points(visualizer, 'LK', traj[4:], color=blue)

        naive_distance = 2*(0.4 + 0.44 + 0.4)   # simply moving straight up, forward, and down
        self.assertTrue(np.linalg.norm(traj) < naive_distance, "Trajectory is longer than 90deg lines")

        shortest_distance = 2*(np.linalg.norm([0.19, 0.4]) +      # go to [0.25, 0.14, 0.4]
                             0.15 +                             # go to [0.4, 0.14, 0.4]
                             np.linalg.norm([0.1, 0.4]) )       # go to [0.5, 0.14, 0.0]
        self.assertTrue(np.linalg.norm(traj) < shortest_distance, "Trajectory is longer than shortest distance")

    def test_visualize_min_rigid_distance_safe_from_file(self):

        # path to file with points to visualize
        points_path = cwd + '/test' + '/poly_min_distance_data.yaml'
        n_frames = 7
        n_boxes = 10
        x_deg = 3

        # collision-free boxes
        box_llim, box_ulim = self.get_sample_collision_free_boxes()

        # Safe boxes
        safe_boxes = OrderedDict()
        safe_boxes['LF'] = fpp.SafeSet(box_llim['LF'], box_ulim['LF'], False)
        safe_boxes['L_knee'] = fpp.SafeSet(box_llim['LF'], box_ulim['LF'], False)

        # Solve minimum distance problem
        traj = np.zeros((70, 3))
        start_idx = 0
        end_idx = x_deg
        with open(points_path, 'r') as f:
            yml = YAML().load(f)
            for p in range(n_frames * n_boxes):
                traj[p, :] = np.array(yml[start_idx:end_idx])
                start_idx += x_deg
                end_idx = start_idx + x_deg

        # visualize safe regions
        traversable_region = OrderedDict()
        traversable_region['LF'] = FrameTraversableRegion('LF', b_visualize_safe=True)
        traversable_region['LF'].load_collision_free_boxes(box_llim['LF'], box_ulim['LF'])
        visualizer = traversable_region['LF'].get_visualizer()

        # visualize points
        LocomanipulationFramePlanner.visualize_simple_points(visualizer, 'torso', traj[0:10])
        LocomanipulationFramePlanner.visualize_simple_points(visualizer, 'LF', traj[10:20])
        LocomanipulationFramePlanner.visualize_simple_points(visualizer, 'RF', traj[20:30])
        LocomanipulationFramePlanner.visualize_simple_points(visualizer, 'LK', traj[30:40], color=blue)
        LocomanipulationFramePlanner.visualize_simple_points(visualizer, 'RK', traj[40:50], color=blue)
        LocomanipulationFramePlanner.visualize_simple_points(visualizer, 'LH', traj[50:60])
        LocomanipulationFramePlanner.visualize_simple_points(visualizer, 'RH', traj[60:70])

        self.assertTrue(True, "Cannot visualize given points")


if __name__ == '__main__':
    unittest.main()
