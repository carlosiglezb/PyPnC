import unittest

import os, sys

from pnc.planner.multicontact.kin_feasibility.self_collision_avoidance.sca_robot_geometry import SCARobotGeometry

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import scipy as sp
import meshcat

from pnc.planner.multicontact.kin_feasibility.multiframe_fpp.mfpp_polygonal import solve_min_reach_iris_distance
from pnc.planner.multicontact.kin_feasibility.multiframe_fpp.mfpp_smooth import optimize_multiple_bezier_iris, \
    optimize_multiple_bezier_iris_casadi, pack_points_for_single_vector
from pnc.planner.multicontact.kin_feasibility.locomanipulation_frame_planner import LocomanipulationFramePlanner
# IRIS
from vision.iris.iris_geom_interface import *
from vision.iris.iris_regions_manager import IrisRegionsManager
# IRIS sequence planner
from pnc.planner.multicontact.kin_feasibility.multiframe_fpp.multiframe_fpp import plan_multistage_iris_seq
from pnc.planner.multicontact.kin_feasibility.planner_surface_contact import PlannerSurfaceContact, MotionFrameSequencer
# Stability polytope tools
from pnc.planner.multicontact.kin_feasibility.stability_polytope_tools import (
    StabilityPolytopeManager, expand_contact_frame_to_points)

b_visualize = True
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

        robot_name = 'g1'   # default, doesn't really matter much
        aux_frames_path = cwd + '/pnc/reachability_map/output/' + robot_name + '/' + \
                               robot_name + '_aux_frames.yaml'
        self.aux_frames = LocomanipulationFramePlanner.add_fixed_distance_between_points(aux_frames_path)

        link_length = self.aux_frames[0]['length']
        self.torso_starting_pos = np.array([0., 0., 0.65])
        self.rf_starting_pos = np.array([-0.2, -0.1, 0.001])
        self.rk_starting_pos = np.array([-0.1, -0.1, np.sqrt(link_length**2 - 0.1**2)])
        self.rh_starting_pos = np.array([-0.2, -0.2, 0.8])
        self.torso_final_pos = np.array([0.2, 0., 0.65])
        self.rf_final_pos = np.array([0.2, -0.1, 0.001])
        self.rk_final_pos = np.array([0.3, -0.1, np.sqrt(link_length**2 - 0.1**2)])
        self.rh_final_pos = np.array([0.2, -0.2, 0.8])
        self.link_length = link_length

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
        motion_frames_seq.add_contact_surfaces([rf_contact_over])

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
        motion_frames_seq.add_contact_surfaces([rf_contact_over])

        # Second sequence: RH
        fixed_frames.append([rf_name])
        motion_frames_seq.add_motion_frame({
                            rh_name: rh_starting_pos + np.array([step_length, 0., 0.])})
        rh_contact_over = PlannerSurfaceContact(rh_name, np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surfaces([rh_contact_over])

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

    def test_multistage_torso_iris_seq_multiple_frame(self):
        # load obstacle, domain, and start / end seed for IRIS
        obstacles = self.obstacles
        domain = self.domain
        rf_starting_pos = self.rf_starting_pos
        torso_starting_pos = self.torso_starting_pos
        rf_ending_pos = self.rf_final_pos
        torso_ending_pos = self.torso_final_pos
        torso_name = 'torso'
        rf_name = 'RF'

        # ------------------- IRIS -------------------
        safe_start_region_rf = IrisGeomInterface(obstacles, domain, rf_starting_pos)
        safe_start_region_torso = IrisGeomInterface(obstacles, domain, torso_starting_pos)
        safe_end_region_rf = IrisGeomInterface(obstacles, domain, rf_ending_pos)
        safe_end_region_torso = IrisGeomInterface(obstacles, domain, torso_ending_pos)
        safe_regions_mgr_dict = {
            torso_name: IrisRegionsManager(safe_start_region_torso, safe_end_region_torso),
            rf_name: IrisRegionsManager(safe_start_region_rf, safe_end_region_rf)
        }
        safe_regions_mgr_dict[torso_name].computeIris()
        safe_regions_mgr_dict[rf_name].computeIris()

        # if start-to-end regions not connected, sample points in between
        safe_regions_mgr_dict[torso_name].connectIrisSeeds()
        safe_regions_mgr_dict[rf_name].connectIrisSeeds()

        if b_visualize:
            # Visualize IRIS regions for "start" and "end" seeds
            safe_regions_mgr_dict[torso_name].visualize(self.vis, torso_name)
            safe_regions_mgr_dict[rf_name].visualize(self.vis, rf_name)

        # ------------------- frame planner -------------------
        step_length = 0.4   # [m]
        fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

        # starting positions for all frames
        starting_pos_dict = {torso_name: torso_starting_pos,
                             rf_name: rf_starting_pos}

        # First sequence: RF
        fixed_frames.append([torso_name])
        motion_frames_seq.add_motion_frame({
                            rf_name: rf_starting_pos + np.array([step_length, 0., 0.])})
        rf_contact_over = PlannerSurfaceContact(rf_name, np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surfaces([rf_contact_over])

        # Second sequence: torso
        fixed_frames.append([rf_name])
        motion_frames_seq.add_motion_frame({
                            torso_name: torso_ending_pos})
        torso_contact_over = PlannerSurfaceContact(rf_name, None)
        motion_frames_seq.add_contact_surfaces([torso_contact_over])

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

        # for the torso we might have two solutions, but most likely to stay in the initial IRIS region
        self.assertTrue(iris_seq[1][torso_name][0] == 0, "torso First box should be the starting position")
        self.assertTrue(sp.linalg.norm(safe_pnt_lst[0][torso_name] - torso_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(safe_pnt_lst[1][torso_name] - torso_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(safe_pnt_lst[2][torso_name] - torso_ending_pos) < 1e-3)

        self.motion_frames_seq = motion_frames_seq
        self.fixed_frames_seq = fixed_frames
        return iris_seq, safe_pnt_lst, safe_regions_mgr_dict

    def test_multistage_iris_seq_rigid_link(self):
        # load obstacle, domain, and start / end seed for IRIS
        obstacles = self.obstacles
        domain = self.domain
        rf_starting_pos = self.rf_starting_pos
        rk_starting_pos = self.rk_starting_pos
        rf_ending_pos = self.rf_final_pos
        rk_ending_pos = self.rk_final_pos
        rf_name = 'RF'
        rk_name = 'R_knee'

        # ------------------- IRIS -------------------
        safe_start_region_rf = IrisGeomInterface(obstacles, domain, rf_starting_pos)
        safe_start_region_rk = IrisGeomInterface(obstacles, domain, rk_starting_pos + np.array([0., 0., -0.1]))
        safe_end_region_rf = IrisGeomInterface(obstacles, domain, rf_ending_pos)
        safe_end_region_rk = IrisGeomInterface(obstacles, domain, rk_ending_pos + np.array([-0.1, 0, -0.1]))
        safe_regions_mgr_dict = {
            rf_name: IrisRegionsManager(safe_start_region_rf, safe_end_region_rf),
            rk_name: IrisRegionsManager(safe_start_region_rk, safe_end_region_rk)}
        safe_regions_mgr_dict[rf_name].computeIris()
        safe_regions_mgr_dict[rk_name].computeIris()

        # if start-to-end regions not connected, sample points in between
        safe_regions_mgr_dict[rf_name].connectIrisSeeds()
        safe_regions_mgr_dict[rk_name].connectIrisSeeds()

        if b_visualize:
            # Visualize IRIS regions for "start" and "end" seeds
            safe_regions_mgr_dict[rf_name].visualize(self.vis, rf_name)
            safe_regions_mgr_dict[rk_name].visualize(self.vis, rk_name)

        # ------------------- frame planner -------------------
        step_length = 0.4   # [m]
        fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

        # starting positions for all frames
        starting_pos_dict = {rf_name: rf_starting_pos,
                             rk_name: rk_starting_pos}

        # First sequence: RF
        fixed_frames.append([])
        motion_frames_seq.add_motion_frame({
                            rf_name: rf_ending_pos,
                            rk_name: rk_ending_pos})
        rf_contact_over = PlannerSurfaceContact(rf_name, np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surfaces([rf_contact_over])

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

        # check on RK
        self.assertTrue(sp.linalg.norm(safe_pnt_lst[0][rk_name] - rk_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(safe_pnt_lst[1][rk_name] - rk_ending_pos) < 1e-3)

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

    def test_min_d_iris_seq_multiple_torso_frame(self):
        iris_seq, safe_points_lst, safe_regions_mgr_dict = self.test_multistage_torso_iris_seq_multiple_frame()

        # test minimum distance method
        reach = None    # ignore reachable space in this test
        traj, length, _ = solve_min_reach_iris_distance(reach, safe_regions_mgr_dict, iris_seq, safe_points_lst)

        traj = np.reshape(traj, [2, 15])
        traj_torso = traj[0].reshape([5, 3])
        traj_rf = traj[1].reshape([5, 3])
        if b_visualize:
            LocomanipulationFramePlanner.visualize_simple_points(self.vis, 'torso/points', traj_torso, [0, 1, 0, 1])
            LocomanipulationFramePlanner.visualize_simple_points(self.vis, 'RF/points', traj_rf, [0, 0, 1, 1])

        self.assertTrue(length < 1e9, "Problem seems infeasible")
        self.assertTrue(sp.linalg.norm(traj_rf[0] - self.rf_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(traj_rf[-1] - self.rf_final_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(traj_torso[0] - self.torso_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(traj_torso[-1] - self.torso_final_pos) < 1e-3)
        self.assertTrue(traj_rf[1][2] > 0.39, "RF-z should be above the knee knocker")
        self.assertTrue(traj_rf[2][2] > 0.39, "RF-z should be above the knee knocker")

    def test_min_d_iris_seq_rigid_link(self):
        iris_seq, safe_points_lst, safe_regions_mgr_dict = self.test_multistage_iris_seq_rigid_link()

        # test minimum distance method
        contact_seq_polygonal = []
        if len(self.fixed_frames_seq[0]) != 0:
            for current_ff_lst in self.fixed_frames_seq:
                if current_ff_lst[0] != 'torso':
                    contact_seq_polygonal.append([current_ff_lst[0]])
                else:
                    contact_seq_polygonal.append([current_ff_lst[1]])
        reach = None    # ignore reachable space in this test
        traj, length, _ = solve_min_reach_iris_distance(reach, safe_regions_mgr_dict, iris_seq,
                                                        safe_points_lst,
                                                        contact_seq=contact_seq_polygonal,
                                                        aux_frames=self.aux_frames)

        traj = np.reshape(traj, [2, 12])
        traj_rf = traj[0].reshape([4, 3])
        traj_rk = traj[1].reshape([4, 3])
        if b_visualize:
            LocomanipulationFramePlanner.visualize_simple_points(self.vis, 'RF/points', traj_rf, [1, 1, 0, 1])
            LocomanipulationFramePlanner.visualize_simple_points(self.vis, 'RK/points', traj_rk, [0, 0, 1, 1])

        self.assertTrue(length < 1e9, "Problem seems infeasible")
        self.assertTrue(sp.linalg.norm(traj_rf[0] - self.rf_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(traj_rf[-1] - self.rf_final_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(traj_rk[0] - self.rk_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(traj_rk[-1] - self.rk_final_pos) < 1e-3)
        self.assertTrue(traj_rf[1][2] > 0.39, "RF-z should be above the knee knocker")
        self.assertTrue(traj_rf[2][2] > 0.39, "RF-z should be above the knee knocker")
        self.assertTrue(sp.linalg.norm(traj_rf[1] - traj_rk[1] - self.link_length) + 0.05)
        self.assertTrue(sp.linalg.norm(traj_rf[2] - traj_rk[2] - self.link_length) + 0.05)

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
        path, sol_stats, _, _ = optimize_multiple_bezier_iris(reach, aux, safe_regions_mgr_dict,
                                                        durations, alpha, safe_points_lst,
                                                        fixed_frames=fixed_frames,
                                                        surface_normals_lst=surface_normals_lst)

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
        path, sol_stats, _, _ = optimize_multiple_bezier_iris(reach, aux, safe_regions_mgr_dict,
                                                        durations, alpha, safe_points_lst,
                                                        fixed_frames=fixed_frames,
                                                        surface_normals_lst=surface_normals_lst)

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

    def test_optimize_bezier_multiple_torso_frame(self):
        iris_seq, safe_points_lst, safe_regions_mgr_dict = self.test_multistage_torso_iris_seq_multiple_frame()
        motion_frames_seq = self.motion_frames_seq
        fixed_frames = self.fixed_frames_seq

        # test optimize multiple bezier
        reach = None    # ignore reachable space in this test
        aux = []
        durations = []        # should be obtained from iris_seq, hard-coded in this test
        durations.append({'torso': np.array([0.2] * 3),
                          'RF': np.array([0.2] * 3)})
        durations.append({'torso': np.array([0.2] * 1),
                          'RF': np.array([0.2] * 1)})
        alpha = {1: 1, 2: 1, 3: 0}
        surface_normals_lst = motion_frames_seq.get_contact_surfaces()
        path, sol_stats, bez_points, dual_vars = optimize_multiple_bezier_iris(reach, aux, safe_regions_mgr_dict,
                                                        durations, alpha, safe_points_lst,
                                                        fixed_frames=fixed_frames,
                                                        surface_normals_lst=surface_normals_lst)
        # path, sol_stats, bez_points, dvars = optimize_multiple_bezier_iris(reach, aux, safe_regions_mgr_dict,
        #                                                 durations, alpha, safe_points_lst,
        #                                                 fixed_frames=fixed_frames,
        #                                                 surface_normals_lst=surface_normals_lst)
        # print(f"Runtime solve with cvxpy: {sol_stats['runtime']}")

        # include simplified rigid bodies for self-collision avoidance
        robot_model_path = cwd + "/robot_model/g1_description/"
        urdf_path = robot_model_path + "g1_cube_collisions.urdf"
        plan_to_model_frames = {
            'torso': 'torso_link',
            'LF': 'left_ankle_roll_link',
            'RF': 'right_ankle_roll_link',
            'L_knee': 'left_knee_link',
            'R_knee': 'right_knee_link',
            'LH': 'left_palm_link',
            'RH': 'right_palm_link'
        }
        sca_geom = SCARobotGeometry(robot_model_path, urdf_path, plan_to_model_frames)
        A1 = sca_geom.get_box_representation('torso')['A']
        b1 = sca_geom.get_box_representation('torso')['b']
        A2 = sca_geom.get_box_representation('RF')['A']
        b2 = sca_geom.get_box_representation('RF')['b']

        # Create points from Bezier curve
        if b_visualize:
            i = 0
            for p in path:
                for seg in range(len(p.beziers)):
                    bezier_curve = [p.beziers[seg]]
                    if i == 0:
                        fr_name = 'torso'
                        A = A1
                        b = b1
                    elif i == 1:
                        fr_name = 'RF'
                        A = A2
                        b = b2
                    LocomanipulationFramePlanner.visualize_bezier_polytope(self.vis, fr_name, bezier_curve, seg, A, b)
                    # LocomanipulationFramePlanner.visualize_bezier_points(self.vis, fr_name, bezier_curve, seg)
                i += 1

            # plot raw bezier points output by casadi
            # for (n_bp, bp_val) in enumerate(bez_points):
            #     if n_bp < 4:
            #         fr_name = 'torso'
            #     elif n_bp < 8:
            #         fr_name = 'RF'
            #     # grab only the position index
            #     LocomanipulationFramePlanner.visualize_simple_points(self.vis, fr_name + "_bez/" + str(n_bp), bp_val[0], [0,0,0,1])

            # plot raw bezier points output by cvxpy
            for n_bp, bp_val in bez_points.items():
                if n_bp < 4:
                    fr_name = 'torso'
                elif n_bp < 8:
                    fr_name = 'RF'
                # grab only the position index
                LocomanipulationFramePlanner.visualize_simple_points(self.vis, fr_name + "_bez/" + str(n_bp), bp_val[0].value, [0,0,0,1])

        self.assertTrue(path is not None, "Problem seems to be infeasible")
        self.assertTrue(sp.linalg.norm(path[0].beziers[0].points[0] - self.torso_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(path[0].beziers[3].points[-1] - self.torso_final_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(path[1].beziers[0].points[0] - self.rf_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(path[1].beziers[3].points[0] - self.rf_final_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(path[1].beziers[-1].points[-1] - self.rf_final_pos) < 1e-3)

        # parse initial guess from previous solution
        bez_initial_guess = {}
        bez_initial_guess['x0'] = pack_points_for_single_vector(bez_points, 'cvxpy')
        bez_initial_guess['lam_g0'] = pack_points_for_single_vector(dual_vars['lam_g0'], 'cvxpy')
        # bez_initial_guess['lam_g0'] = dual_vars['lam_g0']
        bez_initial_guess['lam_x0'] = dual_vars['lam_x0']
        path, sol_stats, bez_points, _ = optimize_multiple_bezier_iris_casadi(reach, aux, safe_regions_mgr_dict,
                                                        durations, alpha, safe_points_lst,
                                                        sca_geom,
                                                        fixed_frames=fixed_frames,
                                                        surface_normals_lst=surface_normals_lst,
                                                        initial_guess=bez_initial_guess)
        # print(f"Runtime solve with CasADi: {sol_stats['runtime']}")

        # Create points from Bezier curve
        if b_visualize:
            i = 0
            for p in path:
                for seg in range(len(p.beziers)):
                    bezier_curve = [p.beziers[seg]]
                    if i == 0:
                        fr_name = 'torso'
                        A = A1
                        b = b1
                    elif i == 1:
                        fr_name = 'RF'
                        A = A2
                        b = b2
                    LocomanipulationFramePlanner.visualize_bezier_polytope(self.vis, fr_name, bezier_curve, seg, A, b)
                    # LocomanipulationFramePlanner.visualize_bezier_points(self.vis, fr_name, bezier_curve, seg)
                i += 1

            # plot raw bezier points output by casadi
            for (n_bp, bp_val) in enumerate(bez_points):
                if n_bp < 4:
                    fr_name = 'torso'
                elif n_bp < 8:
                    fr_name = 'RF'
                # grab only the position index
                LocomanipulationFramePlanner.visualize_simple_points(self.vis, fr_name + "_bez/" + str(n_bp), bp_val[0], [0,0,0,1])

        self.assertTrue(path is not None, "Problem seems to be infeasible")
        error = path[0].beziers[0].points[0] - self.torso_starting_pos
        self.assertTrue(sp.linalg.norm(error) < 1e-3, f"Starting torso distance error: {error}")
        error = path[0].beziers[3].points[-1] - self.torso_final_pos
        self.assertTrue(sp.linalg.norm(error) < 1e-3, f"Final torso distance error: {error}")
        error = path[1].beziers[0].points[0] - self.rf_starting_pos
        self.assertTrue(sp.linalg.norm(error) < 1e-3, f"Starting RF distance error: {error}")
        error = path[1].beziers[3].points[0] - self.rf_final_pos
        self.assertTrue(sp.linalg.norm(error) < 1e-3, f"Final RF distance error: {error}")
        error = path[1].beziers[-1].points[-1] - self.rf_final_pos
        self.assertTrue(sp.linalg.norm(error) < 1e-3, f"Final RF distance error: {error}")


    def test_optimize_bezier_multiple_torso_spheres_frame(self):
        iris_seq, safe_points_lst, safe_regions_mgr_dict = self.test_multistage_torso_iris_seq_multiple_frame()
        motion_frames_seq = self.motion_frames_seq
        fixed_frames = self.fixed_frames_seq

        # test optimize multiple bezier
        reach = None    # ignore reachable space in this test
        aux = []
        durations = []        # should be obtained from iris_seq, hard-coded in this test
        durations.append({'torso': np.array([0.2] * 3),
                          'RF': np.array([0.2] * 3)})
        durations.append({'torso': np.array([0.2] * 1),
                          'RF': np.array([0.2] * 1)})
        alpha = {1: 1, 2: 1, 3: 0}
        surface_normals_lst = motion_frames_seq.get_contact_surfaces()
        path, sol_stats, bez_points, dual_vars = optimize_multiple_bezier_iris(reach, aux, safe_regions_mgr_dict,
                                                        durations, alpha, safe_points_lst,
                                                        fixed_frames=fixed_frames,
                                                        surface_normals_lst=surface_normals_lst)

        # include simplified rigid bodies for self-collision avoidance
        robot_model_path = cwd + "/robot_model/g1_description/"
        # urdf_path = robot_model_path + "g1_cube_sphere_collisions.urdf"
        urdf_path = robot_model_path + "g1_29dof_lock_waist_modified.urdf"
        plan_to_model_frames = {
            'torso': 'torso_link',
            'LF': 'left_ankle_roll_link',
            'RF': 'right_ankle_roll_link',
            'L_knee': 'left_knee_link',
            'R_knee': 'right_knee_link',
            'LH': 'left_palm_link',
            'RH': 'right_palm_link'
        }
        sca_geom = SCARobotGeometry(robot_model_path, urdf_path, plan_to_model_frames)
        # A1 = sca_geom.get_box_representation('torso')['A']
        # b1 = sca_geom.get_box_representation('torso')['b']
        U = sca_geom.get_sphere_representation('RF')['U']

        # Create points from Bezier curve
        if b_visualize:
            i = 0
            for p in path:
                for seg in range(len(p.beziers)):
                    bezier_curve = [p.beziers[seg]]
                    if i == 0:
                        fr_name = 'torso'
                        # LocomanipulationFramePlanner.visualize_bezier_polytope(self.vis, fr_name, bezier_curve, seg, A1, b1)
                    elif i == 1:
                        fr_name = 'RF'
                        radius = 1 / U[0,0]
                        LocomanipulationFramePlanner.visualize_bezier_points(self.vis, fr_name, bezier_curve, seg, radius=radius)
                i += 1

            # plot raw bezier points output by cvxpy
            for n_bp, bp_val in bez_points.items():
                if n_bp < 4:
                    fr_name = 'torso'
                elif n_bp < 8:
                    fr_name = 'RF'
                # grab only the position index
                LocomanipulationFramePlanner.visualize_simple_points(self.vis, fr_name + "_bez/" + str(n_bp), bp_val[0].value, [0,0,0,1])

        self.assertTrue(path is not None, "Problem seems to be infeasible")
        self.assertTrue(sp.linalg.norm(path[0].beziers[0].points[0] - self.torso_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(path[0].beziers[3].points[-1] - self.torso_final_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(path[1].beziers[0].points[0] - self.rf_starting_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(path[1].beziers[3].points[0] - self.rf_final_pos) < 1e-3)
        self.assertTrue(sp.linalg.norm(path[1].beziers[-1].points[-1] - self.rf_final_pos) < 1e-3)

        # parse initial guess from previous solution
        bez_initial_guess = {}
        bez_initial_guess['x0'] = pack_points_for_single_vector(bez_points, 'cvxpy')
        bez_initial_guess['lam_g0'] = pack_points_for_single_vector(dual_vars['lam_g0'], 'cvxpy')
        # bez_initial_guess['lam_g0'] = dual_vars['lam_g0']
        bez_initial_guess['lam_x0'] = dual_vars['lam_x0']
        path, sol_stats, bez_points, _ = optimize_multiple_bezier_iris_casadi(reach, aux, safe_regions_mgr_dict,
                                                        durations, alpha, safe_points_lst,
                                                        sca_geom,
                                                        fixed_frames=fixed_frames,
                                                        surface_normals_lst=surface_normals_lst,
                                                        initial_guess=bez_initial_guess)

        # Create points from Bezier curve
        if b_visualize:
            i = 0
            for p in path:
                for seg in range(len(p.beziers)):
                    bezier_curve = [p.beziers[seg]]
                    if i == 0:
                        fr_name = 'torso'
                        # LocomanipulationFramePlanner.visualize_bezier_polytope(self.vis, fr_name, bezier_curve, seg, A1, b1)
                    elif i == 1:
                        fr_name = 'RF'
                        radius = 1 / U[0,0]
                        LocomanipulationFramePlanner.visualize_bezier_points(self.vis, fr_name, bezier_curve, seg, radius=radius)
                i += 1

            # plot raw bezier points output by casadi
            for (n_bp, bp_val) in enumerate(bez_points):
                if n_bp < 4:
                    fr_name = 'torso'
                elif n_bp < 8:
                    fr_name = 'RF'
                # grab only the position index
                LocomanipulationFramePlanner.visualize_simple_points(self.vis, fr_name + "_bez/" + str(n_bp), bp_val[0], [0,0,0,1])

        self.assertTrue(path is not None, "Problem seems to be infeasible")
        error = path[0].beziers[0].points[0] - self.torso_starting_pos
        self.assertTrue(sp.linalg.norm(error) < 1e-3, f"Starting torso distance error: {error}")
        error = path[0].beziers[3].points[-1] - self.torso_final_pos
        self.assertTrue(sp.linalg.norm(error) < 1e-3, f"Final torso distance error: {error}")
        error = path[1].beziers[0].points[0] - self.rf_starting_pos
        self.assertTrue(sp.linalg.norm(error) < 1e-3, f"Starting RF distance error: {error}")
        error = path[1].beziers[3].points[0] - self.rf_final_pos
        self.assertTrue(sp.linalg.norm(error) < 1e-3, f"Final RF distance error: {error}")
        error = path[1].beziers[-1].points[-1] - self.rf_final_pos
        self.assertTrue(sp.linalg.norm(error) < 1e-3, f"Final RF distance error: {error}")


    # -----------------------------------------------------------------------
    # Stability-polytope tests
    # -----------------------------------------------------------------------

    def test_stability_polytope_manager_direct_constructor(self):
        """
        StabilityPolytopeManager built directly from contact sequences must
        produce valid (non-None) polytopes after compute().
        """
        contact_seqs   = [['RF']]
        contact_planes = [{'RF': np.array([0., 0., 1.])}]
        robot_mass     = 35.0
        manager = StabilityPolytopeManager(
            contact_seqs, contact_planes, robot_mass, n_phases_out=1)

        safe_pnt_lst = [{'RF': self.rf_starting_pos}]
        manager.compute(safe_pnt_lst)

        self.assertTrue(manager.is_computed,
                        "Manager must be marked computed after compute()")
        self.assertEqual(len(manager), 1, "Should have one polytope (one phase)")
        poly = manager.get_polytope(0)
        self.assertIsNotNone(poly, "Phase-0 polytope must not be None for one-foot stance")
        A_stab, b_stab = poly
        self.assertEqual(A_stab.shape[1], 3, "A_stab must have 3 columns")
        self.assertEqual(A_stab.shape[0], b_stab.shape[0],
                         "A_stab rows must match b_stab length")

    def test_stability_polytope_manager_torso_inside(self):
        """
        The torso starting position must lie inside the outer stability
        polytope when both feet are on flat ground.
        """
        contact_seqs   = [['LF', 'RF']]
        contact_planes = [{'LF': np.array([0., 0., 1.]),
                           'RF': np.array([0., 0., 1.])}]
        robot_mass = 35.0
        manager = StabilityPolytopeManager(
            contact_seqs, contact_planes, robot_mass, n_phases_out=1)

        safe_pnt_lst = [{'LF': np.array([-0.2, 0.1, 0.001]),
                         'RF': self.rf_starting_pos}]
        manager.compute(safe_pnt_lst)

        poly = manager.get_polytope(0)
        self.assertIsNotNone(poly, "Polytope must be computed for two-foot stance")

        A_stab, b_stab = poly
        # Check torso start (directly above foot midpoint) is inside A_stab @ p <= b_stab
        p = self.torso_starting_pos
        tol = 0.05   # allow small numerical margin
        self.assertTrue(np.all(A_stab @ p <= b_stab + tol),
                        "Torso starting position should be inside outer stability polytope")

    def test_stability_polytope_manager_from_fixed_frames(self):
        """
        StabilityPolytopeManager.from_fixed_frames() must produce a computed
        manager when fixed_frames contains valid contact-frame names.
        Uses the RF/RH scenario from the multi-frame helper.
        """
        iris_seq, safe_points_lst, safe_regions_mgr_dict = \
            self.test_multistage_iris_seq_multiple_frame()

        robot_mass = 35.0
        # self.fixed_frames_seq = [['RH'], ['RF']] — both are valid contact frames
        manager = StabilityPolytopeManager.from_fixed_frames(
            self.fixed_frames_seq, self.motion_frames_seq, robot_mass)
        manager.compute(safe_points_lst)

        self.assertTrue(manager.is_computed,
                        "from_fixed_frames manager must be computed")
        self.assertGreater(len(manager), 0, "Must have at least one polytope")
        poly = manager.get_polytope(0)
        self.assertIsNotNone(poly, "First polytope must not be None")
        A_stab, b_stab = poly
        self.assertEqual(A_stab.shape[1], 3, "A_stab must be 3-D")

    def test_bezier_opt_lh_rf_stability_polytope_cvxpy(self):
        """
        Cvxpy optimization with LH (left-hand grip) and RF (right foot) as
        fixed bilateral contacts throughout the motion; LF (left foot) crosses
        the knee-knocker sill.  Torso is co-optimised.

        LH and RF enter the optimizer as fixed frames so their trajectories are
        fully constrained to their rest positions, exercising the fixed-frame
        constraint path in optimize_multiple_bezier_iris.  The stability
        polytope is built from LH + RF contacts using 4-corner palm / sole
        expansions.

        Phase 0: LF swings over the sill (torso, LH, RF fixed).
        Phase 1: torso follows (LF, LH, RF fixed).
        """
        # ---- contact-frame positions ----------------------------------------
        lh_pos    = np.array([0., 0.35, 0.9])    # left hand grip, door left post
        lh_normal = np.array([0., -1., 0.])       # surface normal toward robot
        rf_pos    = self.rf_starting_pos           # [-0.2, -0.1, 0.001]
        lf_start  = np.array([-0.2,  0.1, 0.001]) # left foot before sill
        lf_end    = np.array([ 0.2,  0.1, 0.001]) # left foot after sill
        torso_start = self.torso_starting_pos      # [ 0.,   0.,  0.65]
        torso_end   = self.torso_final_pos         # [ 0.2,  0.,  0.65]

        # ---- IRIS for all four frames; LH and RF use same seed (fixed) ------
        obstacles = self.obstacles
        domain    = self.domain
        safe_regions_mgr_dict = {
            'torso': IrisRegionsManager(
                IrisGeomInterface(obstacles, domain, torso_start),
                IrisGeomInterface(obstacles, domain, torso_end)),
            'LF': IrisRegionsManager(
                IrisGeomInterface(obstacles, domain, lf_start),
                IrisGeomInterface(obstacles, domain, lf_end)),
            'LH': IrisRegionsManager(
                IrisGeomInterface(obstacles, domain, lh_pos),
                IrisGeomInterface(obstacles, domain, lh_pos)),
            'RF': IrisRegionsManager(
                IrisGeomInterface(obstacles, domain, rf_pos),
                IrisGeomInterface(obstacles, domain, rf_pos)),
        }
        for mgr in safe_regions_mgr_dict.values():
            mgr.computeIris()
            mgr.connectIrisSeeds()

        if b_visualize:
            for name, mgr in safe_regions_mgr_dict.items():
                mgr.visualize(self.vis, name + '_lhrf')

        # ---- Motion sequence ------------------------------------------------
        fixed_frames      = []
        motion_frames_seq = MotionFrameSequencer()
        starting_pos_dict = {
            'torso': torso_start, 'LF': lf_start,
            'LH': lh_pos,         'RF': rf_pos,
        }

        # Phase 0: LF moves; torso, LH, RF are fixed
        fixed_frames.append(['torso', 'LH', 'RF'])
        motion_frames_seq.add_motion_frame({'LF': lf_end})
        motion_frames_seq.add_contact_surfaces(
            [PlannerSurfaceContact('LF', np.array([0., 0., 1.]))])

        # Phase 1: torso moves; LF, LH, RF are fixed
        fixed_frames.append(['LF', 'LH', 'RF'])
        motion_frames_seq.add_motion_frame({'torso': torso_end})
        motion_frames_seq.add_contact_surfaces(
            [PlannerSurfaceContact('LF', None)])

        motion_frames_lst = motion_frames_seq.get_motion_frames()
        iris_seq, safe_pnt_lst = plan_multistage_iris_seq(
            safe_regions_mgr_dict, fixed_frames, motion_frames_lst, starting_pos_dict)

        # Derive durations directly from the box counts returned by the planner
        durations = [
            {f: np.array([0.2] * len(boxes)) for f, boxes in phase.items()}
            for phase in iris_seq
        ]
        alpha               = {1: 1, 2: 1, 3: 0}
        surface_normals_lst = motion_frames_seq.get_contact_surfaces()

        # ---- Stability polytope: LH (4 palm corners) + RF (4 sole corners) --
        contact_seqs   = [['LH', 'RF'], ['LH', 'RF']]
        contact_planes = [
            {'LH': lh_normal, 'RF': np.array([0., 0., 1.])},
            {'LH': lh_normal, 'RF': np.array([0., 0., 1.])},
        ]
        w_stab       = 1e-2
        stab_manager = StabilityPolytopeManager(
            contact_seqs, contact_planes, 35.0, n_phases_out=len(durations))
        stab_manager.compute(safe_pnt_lst)

        self.assertTrue(stab_manager.is_computed,
                        "Stability manager must be computed for LH+RF contact")
        self.assertIsNotNone(stab_manager.get_polytope(0),
                             "Phase-0 polytope must not be None for LH+RF contact")

        # ---- cvxpy solve with stability-polytope soft constraint ------------
        path, sol_stats, bez_points, _ = optimize_multiple_bezier_iris(
            None, [], safe_regions_mgr_dict, durations, alpha, safe_pnt_lst,
            fixed_frames=fixed_frames,
            surface_normals_lst=surface_normals_lst,
            stab_poly_manager=stab_manager,
            w_stability_polytope=w_stab)

        # ---- visualisation --------------------------------------------------
        if b_visualize:
            frame_names = list(starting_pos_dict.keys())
            for i, fname in enumerate(frame_names):
                for seg in range(len(path[i].beziers)):
                    LocomanipulationFramePlanner.visualize_bezier_points(
                        self.vis, fname + '_lhrf_stab', [path[i].beziers[seg]], seg)
            lh_palm_pts, _ = expand_contact_frame_to_points('LH', lh_pos, lh_normal)
            lh_palm_arr = np.array([p.flatten() for p in lh_palm_pts])
            LocomanipulationFramePlanner.visualize_simple_points(
                self.vis, 'LH_lhrf_stab/palm_corners', lh_palm_arr,
                color=[1., 0., 0., 0.8])
            LocomanipulationFramePlanner.visualize_simple_points(
                self.vis, 'LH_lhrf_stab/center', lh_pos.reshape(1, 3),
                color=[1., 0.5, 0., 1.0])
            rf_sole_pts, _ = expand_contact_frame_to_points(
                'RF', rf_pos, np.array([0., 0., 1.]))
            rf_sole_arr = np.array([p.flatten() for p in rf_sole_pts])
            LocomanipulationFramePlanner.visualize_simple_points(
                self.vis, 'RF_lhrf_stab/sole_corners', rf_sole_arr,
                color=[0., 0., 1., 0.8])
            LocomanipulationFramePlanner.visualize_simple_points(
                self.vis, 'RF_lhrf_stab/center', rf_pos.reshape(1, 3),
                color=[0., 1., 1., 1.0])

        # ---- assertions -----------------------------------------------------
        frame_names = list(starting_pos_dict.keys())
        i_torso = frame_names.index('torso')
        i_lf    = frame_names.index('LF')
        i_lh    = frame_names.index('LH')
        i_rf    = frame_names.index('RF')

        self.assertIsNotNone(path,
                             "cvxpy path must not be None with LH+RF stability constraint")
        self.assertLess(
            sp.linalg.norm(path[i_torso].beziers[0].points[0] - torso_start), 1e-3,
            "Torso must start at torso_start")
        self.assertLess(
            sp.linalg.norm(path[i_torso].beziers[-1].points[-1] - torso_end), 1e-3,
            "Torso must end at torso_end")
        self.assertLess(
            sp.linalg.norm(path[i_lf].beziers[0].points[0] - lf_start), 1e-3,
            "LF must start before the sill")
        self.assertLess(
            sp.linalg.norm(path[i_lf].beziers[-1].points[-1] - lf_end), 1e-3,
            "LF must end after the sill")
        self.assertLess(
            sp.linalg.norm(path[i_lh].beziers[0].points[0] - lh_pos), 1e-3,
            "LH must remain at its grip position (start)")
        self.assertLess(
            sp.linalg.norm(path[i_lh].beziers[-1].points[-1] - lh_pos), 1e-3,
            "LH must remain at its grip position (end)")
        self.assertLess(
            sp.linalg.norm(path[i_rf].beziers[0].points[0] - rf_pos), 1e-3,
            "RF must remain at its standing position (start)")
        self.assertLess(
            sp.linalg.norm(path[i_rf].beziers[-1].points[-1] - rf_pos), 1e-3,
            "RF must remain at its standing position (end)")

    def test_stability_polytope_soft_cost_increases_with_weight(self):
        """
        Raising the stability-polytope weight must not decrease the reported
        cvxpy objective (the soft barrier adds cost, never reduces it).
        Both trajectories must still satisfy the hard boundary constraints.
        """
        iris_seq, safe_points_lst, safe_regions_mgr_dict = \
            self.test_multistage_torso_iris_seq_multiple_frame()
        motion_frames_seq = self.motion_frames_seq
        fixed_frames      = self.fixed_frames_seq

        durations = [
            {'torso': np.array([0.2] * 3), 'RF': np.array([0.2] * 3)},
            {'torso': np.array([0.2] * 1), 'RF': np.array([0.2] * 1)},
        ]
        alpha = {1: 1, 2: 1, 3: 0}
        surface_normals_lst = motion_frames_seq.get_contact_surfaces()

        contact_seqs   = [['RF'], ['RF']]
        contact_planes = [{'RF': np.array([0., 0., 1.])},
                          {'RF': np.array([0., 0., 1.])}]
        stab_manager = StabilityPolytopeManager(
            contact_seqs, contact_planes, 35.0, n_phases_out=len(durations))
        stab_manager.compute(safe_points_lst)

        costs = []
        for w_stab in (0.0, 1e-2):
            _, sol_stats, _, _ = optimize_multiple_bezier_iris(
                None, [], safe_regions_mgr_dict, durations, alpha, safe_points_lst,
                fixed_frames=fixed_frames,
                surface_normals_lst=surface_normals_lst,
                stab_poly_manager=stab_manager if w_stab > 0 else None,
                w_stability_polytope=w_stab)
            costs.append(sol_stats['cost'])

        self.assertGreaterEqual(costs[1], costs[0] - 1e-6,
                                "Stability-polytope soft barrier must not reduce the objective")


if __name__ == '__main__':
    unittest.main()
