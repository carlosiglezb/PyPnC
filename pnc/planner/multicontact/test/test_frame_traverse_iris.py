import unittest
import numpy as np
import scipy as sp
import meshcat

from pnc.planner.multicontact.fastpathplanning.polygonal import solve_min_reach_iris_distance
from pnc.planner.multicontact.fastpathplanning.smooth import optimize_multiple_bezier_iris
from pnc.planner.multicontact.kin_feasibility.locomanipulation_frame_planner import LocomanipulationFramePlanner
# IRIS
from vision.iris.iris_geom_interface import *
from vision.iris.iris_regions_manager import IrisRegionsManager
# IRIS sequence planner
from pnc.planner.multicontact.fastpathplanning.fastpathplanning import plan_multistage_iris_seq
from pnc.planner.multicontact.planner_surface_contact import PlannerSurfaceContact, MotionFrameSequencer

b_visualize = False
b_static_html = False


class TestFrameTraverseIris(unittest.TestCase):
    def setUp(self):
        # create navy door environment
        dom_lb = np.array([-1.6, -0.8, -0.])
        dom_ub = np.array([1.6, 0.8, 2.1])
        floor = mut.HPolyhedron.MakeBox(
                                np.array([-2, -0.9, -0.05]),
                                np.array([2, 0.9, -0.001]))
        knee_knocker_base = mut.HPolyhedron.MakeBox(
                                    np.array([-0.035, -0.9, 0.0]),
                                    np.array([0.035, 0.9, 0.4]))
        knee_knocker_lwall = mut.HPolyhedron.MakeBox(
                                    np.array([-0.035, 0.9-0.518, 0.0]),
                                    np.array([0.035, 0.9, 2.2]))
        knee_knocker_rwall = mut.HPolyhedron.MakeBox(
                                    np.array([-0.035, -0.9, 0.0]),
                                    np.array([0.035, -(0.9-0.518), 2.2]))
        knee_knocker_top = mut.HPolyhedron.MakeBox(
                                    np.array([-0.035, -0.9, 1.85]),
                                    np.array([0.035, 0.9, 2.2]))
        self.obstacles = [floor,
                     knee_knocker_base,
                     knee_knocker_lwall,
                     knee_knocker_rwall,
                     knee_knocker_top]
        self.domain = mut.HPolyhedron.MakeBox(dom_lb, dom_ub)
        self.rf_starting_pos = np.array([-0.2, -0.1, 0.001])
        self.rh_starting_pos = np.array([-0.2, -0.2, 0.8])
        self.rf_final_pos = np.array([0.2, -0.1, 0.001])
        self.rh_final_pos = np.array([0.2, -0.2, 0.8])

        if b_visualize:
            # visualize IRIS region
            self.vis = meshcat.Visualizer()

            # open Visualizer and use default settings
            self.vis.open()
            self.vis.wait()
            self.vis["/Background"].set_property("visible", False)

    def test_multistage_iris_seq_single_frame(self):
        # load obstacle, domain, and start / end seed for IRIS
        obstacles = self.obstacles
        domain = self.domain
        starting_pos = self.rf_starting_pos
        ending_pos = np.array([0.2, -0.1, 0.001])

        # ------------------- IRIS -------------------
        safe_start_region = IrisGeomInterface(obstacles, domain, starting_pos)
        safe_end_region = IrisGeomInterface(obstacles, domain, ending_pos)
        safe_regions_mgr_dict = {'RF': IrisRegionsManager(safe_start_region, safe_end_region)}
        safe_regions_mgr_dict['RF'].computeIris()

        # if start-to-end regions not connected, sample points in between
        if not safe_regions_mgr_dict['RF'].areIrisSeedsContained():
            safe_regions_mgr_dict['RF'].connectIrisSeeds()

        if b_visualize:
            # Visualize IRIS regions for "start" and "end" seeds
            safe_regions_mgr_dict['RF'].visualize(self.vis, 'RF')

        # ------------------- frame planner -------------------
        starting_pos_dict = {'RF': starting_pos}
        step_length = 0.4   # [m]
        motion_frames_seq = MotionFrameSequencer()
        motion_frames_seq.add_motion_frame({
                            'RF': starting_pos + np.array([step_length, 0., 0.])})
        rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(rf_contact_over)

        # plan iris region sequence
        fixed_frames = [None]
        motion_frames_lst = motion_frames_seq.get_motion_frames()
        iris_seq, safe_pnt_lst = plan_multistage_iris_seq(safe_regions_mgr_dict,
                                                         fixed_frames,
                                                         motion_frames_lst,
                                                         starting_pos_dict)

        # check no nan boxes
        for bs in iris_seq:
            for box_idx in bs.values():
                self.assertFalse(np.any(np.isnan(box_idx)),
                                 "Box sequence has unassigned box index in sequence")

        self.assertTrue(iris_seq[0]['RF'][0] == 0, "First box should be the starting position")
        self.assertTrue(iris_seq[0]['RF'][1] == 2, "Second box should be the goal position")
        self.assertTrue(iris_seq[0]['RF'][2] == 1, "Last box should be the created IRIS region")
        self.assertTrue(sp.linalg.norm(safe_pnt_lst[0]['RF'] - starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(safe_pnt_lst[1]['RF'] - ending_pos) < 1e-3)

        self.motion_frames_seq = motion_frames_seq
        return iris_seq, safe_pnt_lst, safe_regions_mgr_dict

    def test_multistage_iris_seq_multiple_frame(self):
        # load obstacle, domain, and start / end seed for IRIS
        obstacles = self.obstacles
        domain = self.domain
        rf_starting_pos = self.rf_starting_pos
        rh_starting_pos = self.rh_starting_pos
        rf_ending_pos = self.rf_final_pos
        rh_ending_pos = self.rh_final_pos
        rf_name = 'RF'
        rh_name = 'RH'

        # ------------------- IRIS -------------------
        safe_start_region_rf = IrisGeomInterface(obstacles, domain, rf_starting_pos)
        safe_start_region_rh = IrisGeomInterface(obstacles, domain, rh_starting_pos)
        safe_end_region_rf = IrisGeomInterface(obstacles, domain, rf_ending_pos)
        safe_end_region_rh = IrisGeomInterface(obstacles, domain, rh_ending_pos)
        safe_regions_mgr_dict = {
            rf_name: IrisRegionsManager(safe_start_region_rf, safe_end_region_rf),
            rh_name: IrisRegionsManager(safe_start_region_rh, safe_end_region_rh)}
        safe_regions_mgr_dict[rf_name].computeIris()
        safe_regions_mgr_dict[rh_name].computeIris()

        # if start-to-end regions not connected, sample points in between
        safe_regions_mgr_dict[rf_name].connectIrisSeeds()
        safe_regions_mgr_dict[rh_name].connectIrisSeeds()

        if b_visualize:
            # Visualize IRIS regions for "start" and "end" seeds
            safe_regions_mgr_dict[rf_name].visualize(self.vis, rf_name)
            safe_regions_mgr_dict[rh_name].visualize(self.vis, rh_name)

        # ------------------- frame planner -------------------
        step_length = 0.4   # [m]
        fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

        # starting positions for all frames
        starting_pos_dict = {rf_name: rf_starting_pos,
                             rh_name: rh_starting_pos}

        # First sequence: RF
        fixed_frames.append([rh_name])
        motion_frames_seq.add_motion_frame({
                            rf_name: rf_starting_pos + np.array([step_length, 0., 0.])})
        rf_contact_over = PlannerSurfaceContact(rf_name, np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(rf_contact_over)

        # Second sequence: RH
        fixed_frames.append([rf_name])
        motion_frames_seq.add_motion_frame({
                            rh_name: rh_starting_pos + np.array([step_length, 0., 0.])})
        rh_contact_over = PlannerSurfaceContact(rh_name, np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(rh_contact_over)

        # plan iris region sequence
        motion_frames_lst = motion_frames_seq.get_motion_frames()
        iris_seq, safe_pnt_lst = plan_multistage_iris_seq(safe_regions_mgr_dict,
                                                         fixed_frames,
                                                         motion_frames_lst,
                                                         starting_pos_dict)

        if b_static_html:
            # create and save locally in static html form
            res = self.vis.static_html()
            save_file = './data/multi-iris-door.html'
            with open(save_file, "w") as f:
                f.write(res)

        # check no nan boxes
        for ir in iris_seq:
            for ir_idx in ir.values():
                self.assertFalse(np.any(np.isnan(ir_idx)),
                                 "Box sequence has unassigned box index in sequence")

        # check box sequence and safe point list are correct
        self.assertTrue(iris_seq[0][rf_name][0] == 0, "RF First Iris region should be the starting position")
        self.assertTrue(iris_seq[0][rf_name][1] == 2, "RF Second Iris region should be the created one")
        self.assertTrue(iris_seq[0][rf_name][2] == 1, "RFLast Iris region should be where the goal is")
        self.assertTrue(sp.linalg.norm(safe_pnt_lst[0][rf_name] - rf_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(safe_pnt_lst[1][rf_name] - rf_ending_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(safe_pnt_lst[2][rf_name] - rf_ending_pos) < 1e-3)

        # for the RH we might have two solutions
        if len(iris_seq[1]) == 2:
            self.assertTrue(iris_seq[1][rh_name][0] == 0, "RH First box should be the starting position")
            self.assertTrue(iris_seq[1][rh_name][1] == 1, "RH Second box must be the ending position")
        elif len(iris_seq[1]) == 3:
            self.assertTrue(iris_seq[1][rh_name][0] == 0, "RH First box should be the starting position")
            self.assertTrue(iris_seq[1][rh_name][1] == 2, "RH Second box must be the created IRIS region")
            self.assertTrue(iris_seq[1][rh_name][2] == 1, "RH Last box must be the ending position")
        self.assertTrue(sp.linalg.norm(safe_pnt_lst[0][rh_name] - rh_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(safe_pnt_lst[1][rh_name] - rh_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(safe_pnt_lst[2][rh_name] - rh_ending_pos) < 1e-3)

        self.motion_frames_seq = motion_frames_seq
        self.fixed_frames_seq = fixed_frames
        return iris_seq, safe_pnt_lst, safe_regions_mgr_dict

    def test_min_d_iris_seq_single_frame(self):
        iris_seq, safe_points_lst, safe_regions_mgr_dict = self.test_multistage_iris_seq_single_frame()

        # test minimum distance method
        reach = None    # ignore reachable space in this test
        traj, length, _ = solve_min_reach_iris_distance(reach, safe_regions_mgr_dict, iris_seq, safe_points_lst)

        traj = np.reshape(traj, [4, 3])
        self.assertTrue(length < 1e9, "Problem seems infeasible")
        self.assertTrue(sp.linalg.norm(traj[0] - self.rf_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(traj[-1] - self.rf_final_pos) < 1e-3)

    def test_min_d_iris_seq_multiple_frame(self):
        iris_seq, safe_points_lst, safe_regions_mgr_dict = self.test_multistage_iris_seq_multiple_frame()

        # test minimum distance method
        reach = None    # ignore reachable space in this test
        traj, length, _ = solve_min_reach_iris_distance(reach, safe_regions_mgr_dict, iris_seq, safe_points_lst)

        traj = np.reshape(traj, [2, 18])
        traj_rf = traj[0].reshape([6, 3])
        traj_rh = traj[1].reshape([6, 3])
        if b_visualize:
            LocomanipulationFramePlanner.visualize_simple_points(self.vis, 'RF/points', traj_rf, [0, 0, 1, 1])
            LocomanipulationFramePlanner.visualize_simple_points(self.vis, 'RH/points', traj_rh, [0, 0, 1, 1])

        self.assertTrue(length < 1e9, "Problem seems infeasible")
        self.assertTrue(sp.linalg.norm(traj_rf[0] - self.rf_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(traj_rf[-1] - self.rf_final_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(traj_rh[0] - self.rh_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(traj_rh[-1] - self.rh_final_pos) < 1e-3)
        self.assertTrue(traj_rf[1][2] > 0.39, "RF-z should be above the knee knocker")
        self.assertTrue(traj_rf[2][2] > 0.39, "RF-z should be above the knee knocker")

    def test_optimize_bezier_single_frame(self):
        iris_seq, safe_points_lst, safe_regions_mgr_dict = self.test_multistage_iris_seq_single_frame()
        motion_frames_seq = self.motion_frames_seq

        # test optimize multiple bezier
        fixed_frames = [None]     # one per segment
        reach = None    # ignore reachable space in this test
        aux = []
        durations=[]        # should be obtained from iris_seq, hard-coded in this test
        durations.append({'RF': np.array([0.2] * 3)})
        alpha = {1: 1, 2: 2, 3: 0.1}
        surface_normals_lst = motion_frames_seq.get_contact_surfaces()
        path, sol_stats, _ = optimize_multiple_bezier_iris(reach, aux, safe_regions_mgr_dict,
                                                        durations, alpha, safe_points_lst,
                                                        fixed_frames, surface_normals_lst)

        # Visualize points from Bezier curve
        if b_visualize:
            for p in path:
                for seg in range(len(p.beziers)):
                    bezier_curve = [p.beziers[seg]]
                    fr_name = 'RF'
                    LocomanipulationFramePlanner.visualize_bezier_points(self.vis, fr_name, bezier_curve, seg)

        self.assertTrue(path is not None, "Problem seems to be infeasible")
        self.assertTrue(sp.linalg.norm(path[0].beziers[0].points[0] - self.rf_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(path[0].beziers[2].points[-1] - self.rf_final_pos) < 1e-3)

    def test_optimize_bezier_multiple_frame(self):
        iris_seq, safe_points_lst, safe_regions_mgr_dict = self.test_multistage_iris_seq_multiple_frame()
        motion_frames_seq = self.motion_frames_seq
        fixed_frames = self.fixed_frames_seq

        # test optimize multiple bezier
        reach = None    # ignore reachable space in this test
        aux = []
        durations=[]        # should be obtained from iris_seq, hard-coded in this test
        durations.append({'RF': np.array([0.2] * 3),
                          'RH': np.array([0.2] * 3)})
        durations.append({'RF': np.array([0.3] * 2),
                          'RH': np.array([0.3] * 2)})
        alpha = {1: 1, 2: 1, 3: 0.1}
        surface_normals_lst = motion_frames_seq.get_contact_surfaces()
        path, sol_stats, _ = optimize_multiple_bezier_iris(reach, aux, safe_regions_mgr_dict,
                                                        durations, alpha, safe_points_lst,
                                                        fixed_frames, surface_normals_lst)

        # Create points from Bezier curve
        if b_visualize:
            i = 0
            for p in path:
                for seg in range(len(p.beziers)):
                    bezier_curve = [p.beziers[seg]]
                    if i == 0:
                        fr_name = 'RF'
                    elif i == 1:
                        fr_name = 'RH'
                    LocomanipulationFramePlanner.visualize_bezier_points(self.vis, fr_name, bezier_curve, seg)
                i += 1

        self.assertTrue(path is not None, "Problem seems to be infeasible")
        self.assertTrue(sp.linalg.norm(path[0].beziers[0].points[0] - self.rf_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(path[0].beziers[2].points[-1] - self.rf_final_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(path[1].beziers[0].points[0] - self.rh_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(path[1].beziers[2].points[0] - self.rh_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(path[1].beziers[-1].points[-1] - self.rh_final_pos) < 1e-2)

if __name__ == '__main__':
    unittest.main()
