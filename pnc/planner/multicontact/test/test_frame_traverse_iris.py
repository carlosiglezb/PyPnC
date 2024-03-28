import unittest
import numpy as np
import meshcat

# IRIS
from vision.iris.iris_geom_interface import *
from vision.iris.iris_regions_manager import IrisRegionsManager
# IRIS sequence planner
from pnc.planner.multicontact.fastpathplanning.fastpathplanning import plan_multistage_iris_seq
from pnc.planner.multicontact.planner_surface_contact import PlannerSurfaceContact, MotionFrameSequencer

b_visualize = False


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
        self.starting_pos = np.array([-0.2, -0.1, 0.001])

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
        starting_pos = self.starting_pos
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
            safe_regions_mgr_dict['RF'].visualize(self.vis)

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
        box_seq, safe_pnt_lst = plan_multistage_iris_seq(safe_regions_mgr_dict,
                                                         fixed_frames,
                                                         motion_frames_lst,
                                                         starting_pos_dict)

        # check no nan boxes
        for bs in box_seq:
            for box_idx in bs.values():
                self.assertFalse(np.any(np.isnan(box_idx)),
                                 "Box sequence has unassigned box index in sequence")

        self.assertTrue(box_seq[0]['RF'][0] == 0, "First box should be the starting position")
        self.assertTrue(box_seq[0]['RF'][1] == 2, "Second box should be the goal position")
        self.assertTrue(box_seq[0]['RF'][2] == 1, "Last box should be the created IRIS region")

if __name__ == '__main__':
    unittest.main()
