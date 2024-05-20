import os
import sys
from collections import OrderedDict

import crocoddyl
from pnc.planner.multicontact.crocoddyl.ConstraintModelRCJ import ConstraintModelRCJ
# Collision free description
from pydrake.geometry.optimization import HPolyhedron

from pnc.planner.multicontact.crocoddyl.ResidualModelStateError import ResidualModelStateError
# Kinematic feasibility
from pnc.planner.multicontact.kin_feasibility.frame_traversable_region import FrameTraversableRegion
from pnc.planner.multicontact.planner_surface_contact import PlannerSurfaceContact, MotionFrameSequencer
from pnc.planner.multicontact.kin_feasibility.ik_cfree_planner import *
# Tools for dynamic feasibility
from humanoid_action_models import *

# Visualization tools
import matplotlib.pyplot as plt
from plot.helper import plot_vector_traj, Fxyz_labels
import plot.meshcat_utils as vis_tools
from vision.iris.iris_regions_manager import IrisRegionsManager, IrisGeomInterface

cwd = os.getcwd()
sys.path.append(cwd)

B_SHOW_JOINT_PLOTS = True
B_SHOW_GRF_PLOTS = False
B_VISUALIZE = True


def get_draco3_shaft_wrist_default_initial_pose():
    q0 = np.zeros(27, )
    hip_yaw_angle = 5
    q0[0] = 0.  # l_hip_ie
    q0[1] = np.radians(hip_yaw_angle)  # l_hip_aa
    q0[2] = -np.pi / 4  # l_hip_fe
    q0[3] = np.pi / 4  # l_knee_fe_jp
    q0[4] = np.pi / 4  # l_knee_fe_jd
    q0[5] = -np.pi / 4  # l_ankle_fe
    q0[6] = np.radians(-hip_yaw_angle)  # l_ankle_ie
    q0[7] = 0.  # l_shoulder_fe
    q0[8] = np.pi / 6  # l_shoulder_aa
    q0[9] = 0.  # l_shoulder_ie
    q0[10] = -np.pi / 2  # l_elbow_fe
    q0[11] = -np.pi/3.  # l_wrist_ps
    q0[12] = 0.  # l_wrist_pitch
    q0[13] = 0.  # neck pitch
    q0[14] = 0.  # r_hip_ie
    q0[15] = np.radians(-hip_yaw_angle)  # r_hip_aa
    q0[16] = -np.pi / 4  # r_hip_fe
    q0[17] = np.pi / 4  # r_knee_fe_jp
    q0[18] = np.pi / 4  # r_knee_fe_jd
    q0[19] = -np.pi / 4  # r_ankle_fe
    q0[20] = np.radians(hip_yaw_angle)  # r_ankle_ie
    q0[21] = 0.  # r_shoulder_fe
    q0[22] = -np.pi / 6  # r_shoulder_aa
    q0[23] = 0.  # r_shoulder_ie
    q0[24] = -np.pi / 2  # r_elbow_fe
    q0[25] = np.pi/3.   # r_wrist_ps
    q0[26] = 0.  # r_wrist_pitch

    floating_base = np.array([0., 0., 0.741, 0., 0., 0., 1.])
    return np.concatenate((floating_base, q0))


def load_navy_env():
    # create navy door environment
    door_pos = np.array([0.32, 0., 0.])
    door_quat = np.array([0., 0., 0.7071068, 0.7071068])
    door_width = np.array([0.025, 0., 0.])
    dom_ubody_lb = np.array([-1.6, -0.8, 0.5])
    dom_ubody_ub = np.array([1.6, 0.8, 2.1])
    dom_lbody_lb = np.array([-1.6, -0.8, -0.])
    dom_lbody_ub = np.array([1.6, 0.8, 1.1])
    floor = HPolyhedron.MakeBox(
        np.array([-2, -0.9, -0.05]) + door_pos + door_width,
        np.array([2, 0.9, -0.001]) + door_pos + door_width)
    knee_knocker_base = HPolyhedron.MakeBox(
        np.array([-0.025, -0.9, 0.0]) + door_pos + door_width,
        np.array([0.025, 0.9, 0.4]) + door_pos + door_width)
    knee_knocker_lwall = HPolyhedron.MakeBox(
        np.array([-0.025, 0.9 - 0.518, 0.0]) + door_pos + door_width,
        np.array([0.025, 0.9, 2.2]) + door_pos + door_width)
    knee_knocker_rwall = HPolyhedron.MakeBox(
        np.array([-0.025, -0.9, 0.0]) + door_pos + door_width,
        np.array([0.025, -(0.9 - 0.518), 2.2]) + door_pos + door_width)
    knee_knocker_top = HPolyhedron.MakeBox(
        np.array([-0.025, -0.9, 1.85]) + door_pos + door_width,
        np.array([0.025, 0.9, 2.25]) + door_pos + door_width)
    knee_knocker_llip = HPolyhedron.MakeBox(
        np.array([-0.035, 0.9 - 0.518, 0.25]) + door_pos + door_width,
        np.array([0.035, 0.9 - 0.518 + 0.15, 2.0]) + door_pos + door_width)
    knee_knocker_rlip = HPolyhedron.MakeBox(
        np.array([-0.035, -(0.9 - 0.518 + 0.15), 0.25]) + door_pos + door_width,
        np.array([0.035, -(0.9 - 0.518), 2.0]) + door_pos + door_width)
    obstacles = [floor,
                      knee_knocker_base,
                      knee_knocker_lwall,
                      knee_knocker_rwall,
                      knee_knocker_llip,
                      knee_knocker_rlip,
                      knee_knocker_top]
    domain_ubody = HPolyhedron.MakeBox(dom_ubody_lb, dom_ubody_ub)
    domain_lbody = HPolyhedron.MakeBox(dom_lbody_lb, dom_lbody_ub)

    door_pose = np.concatenate((door_pos, door_quat))
    return door_pose, obstacles, domain_ubody, domain_lbody


def load_robot_model():
    draco_urdf_file = cwd + "/robot_model/draco3/draco3_ft_wrist_mesh_updated.urdf"
    package_dir = cwd + "/robot_model/draco3"
    rob_model, col_model, vis_model = pin.buildModelsFromUrdf(draco_urdf_file,
                                                              package_dir,
                                                              pin.JointModelFreeFlyer())
    rob_data, col_data, vis_data = pin.createDatas(rob_model, col_model, vis_model)

    return rob_model, col_model, vis_model, rob_data, col_data, vis_data


def compute_iris_regions_mgr(obstacles,
                             domain_ubody,
                             domain_lbody,
                             robot_data,
                             plan_to_model_ids,
                             standing_pos,
                             goal_step_length):
    # shift (feet) iris seed to get nicer IRIS region
    iris_lf_shift = np.array([0.1, 0., 0.])
    iris_rf_shift = np.array([0.1, 0., 0.])
    iris_kn_shift = np.array([-0.08, 0., 0.04])

    # get end effector positions via fwd kin
    starting_torso_pos = standing_pos
    final_torso_pos = starting_torso_pos + np.array([goal_step_length, 0., 0.])
    starting_lf_pos = robot_data.oMf[plan_to_model_ids['LF']].translation
    final_lf_pos = starting_lf_pos + np.array([goal_step_length, 0., 0.])
    starting_lh_pos = robot_data.oMf[plan_to_model_ids['LH']].translation - np.array([0.01, 0., 0.])
    final_lh_pos = starting_lh_pos + np.array([goal_step_length, 0., 0.])
    starting_rf_pos = robot_data.oMf[plan_to_model_ids['RF']].translation
    final_rf_pos = starting_rf_pos + np.array([goal_step_length, 0., 0.])
    starting_rh_pos = robot_data.oMf[plan_to_model_ids['RH']].translation - np.array([0.01, 0., 0.])
    final_rh_pos = starting_rh_pos + np.array([goal_step_length + 0.15, 0., 0.])
    starting_lkn_pos = robot_data.oMf[plan_to_model_ids['L_knee']].translation
    final_lkn_pos = starting_lkn_pos + np.array([goal_step_length, 0., 0.])
    starting_rkn_pos = robot_data.oMf[plan_to_model_ids['R_knee']].translation
    final_rkn_pos = starting_rkn_pos + np.array([goal_step_length, 0., 0.])

    safe_torso_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_torso_pos)
    safe_torso_end_region = IrisGeomInterface(obstacles, domain_ubody, final_torso_pos)
    safe_lf_start_region = IrisGeomInterface(obstacles, domain_lbody, starting_lf_pos + iris_lf_shift)
    safe_lf_end_region = IrisGeomInterface(obstacles, domain_lbody, final_lf_pos)
    safe_lk_start_region = IrisGeomInterface(obstacles, domain_lbody, starting_lkn_pos)
    safe_lk_end_region = IrisGeomInterface(obstacles, domain_lbody, final_lkn_pos + iris_kn_shift)
    safe_lh_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_lh_pos)
    safe_lh_end_region = IrisGeomInterface(obstacles, domain_ubody, final_lh_pos)
    safe_rf_start_region = IrisGeomInterface(obstacles, domain_lbody, starting_rf_pos + iris_rf_shift)
    safe_rf_end_region = IrisGeomInterface(obstacles, domain_lbody, final_rf_pos)
    safe_rk_start_region = IrisGeomInterface(obstacles, domain_lbody, starting_rkn_pos)
    safe_rk_end_region = IrisGeomInterface(obstacles, domain_lbody, final_rkn_pos + iris_kn_shift)
    safe_rh_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_rh_pos)
    safe_rh_end_region = IrisGeomInterface(obstacles, domain_ubody, final_rh_pos)
    safe_regions_mgr_dict = {'torso': IrisRegionsManager(safe_torso_start_region, safe_torso_end_region),
                             'LF': IrisRegionsManager(safe_lf_start_region, safe_lf_end_region),
                             'L_knee': IrisRegionsManager(safe_lk_start_region, safe_lk_end_region),
                             'LH': IrisRegionsManager(safe_lh_start_region, safe_lh_end_region),
                             'RF': IrisRegionsManager(safe_rf_start_region, safe_rf_end_region),
                             'R_knee': IrisRegionsManager(safe_rk_start_region, safe_rk_end_region),
                             'RH': IrisRegionsManager(safe_rh_start_region, safe_rh_end_region)}

    # compute and connect IRIS from start to goal
    for _, irm in safe_regions_mgr_dict.items():
        irm.computeIris()
        irm.connectIrisSeeds()

    # save initial/final EE positions
    p_init = {}
    p_init['torso'] = starting_torso_pos
    p_init['LF'] = starting_lf_pos
    p_init['RF'] = starting_rf_pos
    p_init['L_knee'] = starting_lkn_pos
    p_init['R_knee'] = starting_rkn_pos
    p_init['LH'] = starting_lh_pos
    p_init['RH'] = starting_rh_pos

    return safe_regions_mgr_dict, p_init


def get_five_stage_one_hand_contact_sequence(safe_regions_mgr_dict):
    ###### Previously used key locations
    # door_l_outer_location = np.array([0.45, 0.35, 1.2])
    # door_r_outer_location = np.array([0.45, -0.35, 1.2])
    # door_l_inner_location = np.array([0.52, 0.3, 1.2])
    # door_r_inner_location = np.array([0.52, -0.3, 1.2])

    starting_lh_pos = safe_regions_mgr_dict['LH'].iris_list[0].seed_pos
    starting_rh_pos = safe_regions_mgr_dict['RH'].iris_list[0].seed_pos
    final_lf_pos = safe_regions_mgr_dict['LF'].iris_list[1].seed_pos
    final_lkn_pos = safe_regions_mgr_dict['L_knee'].iris_list[1].seed_pos
    final_rf_pos = safe_regions_mgr_dict['RF'].iris_list[1].seed_pos
    final_torso_pos = safe_regions_mgr_dict['torso'].iris_list[1].seed_pos
    final_rkn_pos = safe_regions_mgr_dict['R_knee'].iris_list[1].seed_pos
    final_rh_pos = safe_regions_mgr_dict['RH'].iris_list[1].seed_pos

    # initialize fixed and motion frame sets
    fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

    # ---- Step 1: L hand to frame
    fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'RH'])   # frames that must not move
    motion_frames_seq.add_motion_frame({'LH': starting_lh_pos + np.array([0.08, 0.07, 0.15])})
    lh_contact_front = PlannerSurfaceContact('LH', np.array([-1, 0, 0]))
    lh_contact_front.set_contact_breaking_velocity(np.array([-1, 0., 0.]))
    motion_frames_seq.add_contact_surface(lh_contact_front)

    # ---- Step 2: step through door with left foot
    fixed_frames.append(['RF', 'R_knee', 'LH'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
                        'LF': final_lf_pos,
                        'L_knee': final_lkn_pos})
    lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surface(lf_contact_over)

    # ---- Step 3: re-position L/R hands for more stability
    fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
                        'LH': starting_lh_pos + np.array([0.3, 0.0, 0.0])})
                        # 'RH': starting_rh_pos + np.array([0.08, -0.07, 0.15])})
    lh_contact_inside = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
    lh_contact_inside.set_contact_breaking_velocity(np.array([-1, 0., 0.]))
    # rh_contact_inside = PlannerSurfaceContact('RH', np.array([0, 1, 0]))
    motion_frames_seq.add_contact_surface([lh_contact_inside])

    # ---- Step 4: step through door with right foot
    fixed_frames.append(['LF', 'L_knee'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
                        'RF': final_rf_pos,
                        'torso': final_torso_pos,
                        'R_knee': final_rkn_pos,
                        'RH': final_rh_pos,
                        'LH': starting_lh_pos + np.array([0.55, 0.0, 0.0])})
    rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surface(rf_contact_over)

    # ---- Step 5: square up
    fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
    motion_frames_seq.add_motion_frame({})

    return fixed_frames, motion_frames_seq


def visualize_env(rob_model, rob_collision_model, rob_visual_model, q0, door_pose):
    # visualize robot and door
    visualizer = MeshcatVisualizer(rob_model, rob_collision_model, rob_visual_model)

    try:
        visualizer.initViewer(open=True)
        visualizer.viewer.wait()
    except ImportError as err:
        print(
            "Error while initializing the viewer. It seems you should install Python meshcat"
        )
        print(err)
        sys.exit(0)
    visualizer.loadViewerModel(rootNodeName="draco3")
    visualizer.display(q0)

    # load (real) door to visualizer
    door_model, door_collision_model, door_visual_model = pin.buildModelsFromUrdf(
        cwd + "/robot_model/ground/navy_door.urdf",
        cwd + "/robot_model/ground", pin.JointModelFreeFlyer())

    door_vis = MeshcatVisualizer(door_model, door_collision_model, door_visual_model)
    door_vis.initViewer(visualizer.viewer)
    door_vis.loadViewerModel(rootNodeName="door")
    door_vis_q = door_pose
    door_vis.display(door_vis_q)

    return visualizer


def main(args):
    contact_seq = args.sequence

    #
    # Initialize frames to consider for contact planning
    #
    plan_to_model_frames = OrderedDict()
    plan_to_model_frames['torso'] = 'torso_link'
    plan_to_model_frames['LF'] = 'l_foot_contact'
    plan_to_model_frames['RF'] = 'r_foot_contact'
    plan_to_model_frames['L_knee'] = 'l_knee_fe_ld'
    plan_to_model_frames['R_knee'] = 'r_knee_fe_ld'
    plan_to_model_frames['LH'] = 'l_hand_contact'
    plan_to_model_frames['RH'] = 'r_hand_contact'

    #
    # Load robot model, reachable regions, and environment
    #
    aux_frames_path = cwd + '/pnc/reachability_map/output/draco3_aux_frames.yaml'
    ee_halfspace_params, frame_stl_paths = OrderedDict(), OrderedDict()
    for fr in plan_to_model_frames.keys():
        ee_halfspace_params[fr] = cwd + '/pnc/reachability_map/output/draco3_' + fr + '.yaml'
        frame_stl_paths[fr] = (cwd + '/pnc/reachability_map/output/draco3_' + fr + '.stl')

    door_pose, obstacles, domain_ubody, domain_lbody = load_navy_env()
    rob_model, col_model, vis_model, rob_data, col_data, vis_data = load_robot_model()
    q0 = get_draco3_shaft_wrist_default_initial_pose()
    v0 = np.zeros(rob_model.nv)
    x0 = np.concatenate([q0, v0])

    # Update Pinocchio model
    pin.forwardKinematics(rob_model, rob_data, q0)
    pin.updateFramePlacements(rob_model, rob_data)

    # Getting the frame ids
    plan_to_model_ids = {}
    plan_to_model_ids['RF'] = rob_model.getFrameId(plan_to_model_frames['RF'])
    plan_to_model_ids['LF'] = rob_model.getFrameId(plan_to_model_frames['LF'])
    plan_to_model_ids['R_knee'] = rob_model.getFrameId(plan_to_model_frames['R_knee'])
    plan_to_model_ids['L_knee'] = rob_model.getFrameId(plan_to_model_frames['L_knee'])
    plan_to_model_ids['LH'] = rob_model.getFrameId(plan_to_model_frames['LH'])
    plan_to_model_ids['RH'] = rob_model.getFrameId(plan_to_model_frames['RH'])
    plan_to_model_ids['torso'] = rob_model.getFrameId(plan_to_model_frames['torso'])
    rf_id = rob_model.getFrameId(plan_to_model_frames['RF'])
    lf_id = rob_model.getFrameId(plan_to_model_frames['LF'])
    lh_id = rob_model.getFrameId(plan_to_model_frames['LH'])
    rh_id = rob_model.getFrameId(plan_to_model_frames['RH'])
    base_id = rob_model.getFrameId(plan_to_model_frames['torso'])

    # Generate IRIS regions
    standing_pos = q0[:3]
    step_length = 0.35
    safe_regions_mgr_dict, p_init = compute_iris_regions_mgr(obstacles, domain_ubody, domain_lbody,
                                                     rob_data, plan_to_model_ids,
                                                     standing_pos, step_length)

    if B_VISUALIZE:
        visualizer = visualize_env(rob_model, col_model, vis_model, q0, door_pose)
    else:
        visualizer = None

    #
    # Initialize IK Frame Planner
    #
    ik_cfree_planner = IKCFreePlanner(rob_model, rob_data, plan_to_model_frames, q0)

    # generate all frame traversable regions
    traversable_regions_dict = OrderedDict()
    for fr in plan_to_model_frames.keys():
        if fr == 'torso':
            traversable_regions_dict[fr] = FrameTraversableRegion(fr,
                                                                  b_visualize_reach=B_VISUALIZE,
                                                                  b_visualize_safe=B_VISUALIZE,
                                                                  visualizer=visualizer)
        else:
            traversable_regions_dict[fr] = FrameTraversableRegion(fr,
                                                                  frame_stl_paths[fr],
                                                                  ee_halfspace_params[fr],
                                                                  b_visualize_reach=B_VISUALIZE,
                                                                  b_visualize_safe=B_VISUALIZE,
                                                                  visualizer=visualizer)
            traversable_regions_dict[fr].update_origin_pose(standing_pos)
        traversable_regions_dict[fr].load_iris_regions(safe_regions_mgr_dict[fr])

    # hand-chosen five-stage sequence of contacts
    fixed_frames_seq, motion_frames_seq = get_five_stage_one_hand_contact_sequence(safe_regions_mgr_dict)

    # planner parameters
    T = 3
    alpha = [0, 0, 1]
    traversable_regions = [traversable_regions_dict['torso'],
                           traversable_regions_dict['LF'],
                           traversable_regions_dict['RF'],
                           traversable_regions_dict['L_knee'],
                           traversable_regions_dict['R_knee'],
                           traversable_regions_dict['LH'],
                           traversable_regions_dict['RH']]
    frame_planner = LocomanipulationFramePlanner(traversable_regions,
                                                 aux_frames_path=aux_frames_path,
                                                 fixed_frames=fixed_frames_seq,
                                                 motion_frames_seq=motion_frames_seq)

    # compute paths and create targets
    ik_cfree_planner.set_planner(frame_planner)
    ik_cfree_planner.plan(p_init, T, alpha, visualizer)

    #
    # Start Dynamic Feasibility Check
    #
    n_q = len(q0)
    l_constr_ids, r_constr_ids = [9 + n_q, 10 + n_q], [23 + n_q, 24 + n_q]    # qdot
    l_constr_ids_u, r_constr_ids_u = [3, 4], [17, 18]       # u
    state = crocoddyl.StateMultibody(rob_model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)

    constr_mgr = crocoddyl.ConstraintModelManager(state, actuation.nu)
    # -------- Existent constraint --------
    # res_model = crocoddyl.ResidualModelState(state, x0, actuation.nu)
    # constr_model_res = crocoddyl.ConstraintModelResidual(state, res_model)
    # constr_mgr.addConstraint("residual_model", constr_model_res)
    # -------- New constraint --------
    l_res_model = ResidualModelStateError(state, 1, nu=actuation.nu, q_dependent=False)
    l_res_model.constr_ids = l_constr_ids
    # l_res_model.constr_ids_u = l_constr_ids_u
    l_rcj_constr = ConstraintModelRCJ(state, residual=l_res_model, ng=0, nh=1)
    constr_mgr.addConstraint("l_rcj_constr", l_rcj_constr)
    r_res_model = ResidualModelStateError(state, 1, nu=actuation.nu, q_dependent=False)
    r_res_model.constr_ids = r_constr_ids
    # r_res_model.constr_ids_u = r_constr_ids_u
    r_rcj_constr = ConstraintModelRCJ(state, residual=r_res_model, ng=0, nh=1)
    constr_mgr.addConstraint("r_rcj_constr", r_rcj_constr)

    # ---------- testing ----------
    # shared_data = crocoddyl.DataCollectorMultibody(rob_data)
    # data = constr_mgr.createData(shared_data)
    # constr_mgr.calc(data, x0, np.zeros(actuation.nu))

    #
    # Dynamic solve
    #
    DT = 2e-2

    # Connecting the sequences of models
    # NUM_OF_CONTACT_CONFIGURATIONS = len(motion_frames_seq.motion_frame_lst)
    NUM_OF_CONTACT_CONFIGURATIONS = 2
    T_total = T * NUM_OF_CONTACT_CONFIGURATIONS

    lh_targets, base_into_targets, lf_targets = [], [], []
    # for left_t, right_t in zip(lh_targets, rh_targets):
    #     dmodel = createDoubleSupportActionModel(left_t, right_t)
    #     model_seqs += createSequence([dmodel], DT, N)

    # Defining the problem and the solver
    fddp = [crocoddyl.SolverFDDP] * NUM_OF_CONTACT_CONFIGURATIONS
    for i in range(NUM_OF_CONTACT_CONFIGURATIONS):
        model_seqs = []
        if i == 0:
            # Reach door with left hand
            N_lhand_to_door = 20  # knots for left hand reaching
            # for lhand_t in lh_targets:
            for t in np.linspace(i*T, (i+1)*T, N_lhand_to_door):
                lhand_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('LH'), t)
                dmodel = createDoubleSupportActionModel(state,
                                                        actuation,
                                                        rob_data,
                                                        x0,
                                                        lf_id,
                                                        rf_id,
                                                        lh_id,
                                                        rh_id,
                                                        constr_mgr,
                                                        lh_target=lhand_t)
                lh_targets.append(lhand_t)
                model_seqs += createSequence([dmodel], DT, 1)

        elif i == 1:
            plan_to_model_ids.pop('L_knee')
            plan_to_model_ids.pop('R_knee')
            # DT = 0.015
            # Using left-hand support, pass left-leg through door
            ee_rpy = {'LH': [0., -np.pi/2, 0.]}
            N_base_through_door = 60  # knots per waypoint to pass through door
            # for base_t, lfoot_t in zip(base_into_targets, lf_targets):
            for t in np.linspace(i*T, (i+1)*T, N_base_through_door):
                lfoot_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('LF'), t)
                # lknee_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('L_knee'), t)
                rfoot_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('RF'), t)
                # rknee_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('R_knee'), t)
                lhand_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('LH'), t)
                rhand_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('RH'), t)
                base_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('torso'), t)
                frame_targets_dict = {
                    'torso': base_t,
                    'LF': lfoot_t,
                    'RF': rfoot_t,
                    # 'L_knee': lknee_t,
                    # 'R_knee': rknee_t,
                    'LH': lhand_t,
                    'RH': rhand_t
                }
                dmodel = createMultiFrameActionModel(state,
                                                     actuation,
                                                     x0,
                                                     plan_to_model_ids,
                                                     ['LH', 'RF'],
                                                     ee_rpy,
                                                     frame_targets_dict,
                                                     constr_mgr)
                base_into_targets.append(base_t)
                lf_targets.append(lfoot_t)
                model_seqs += createSequence([dmodel], DT, 1)

        # elif i == 2:
        #     DT = 0.02
        #     # Reach door with left and right hand from inside
        #     N_rhand_to_door = 20  # knots for left hand reaching
        #     for lhand_t, rhand_t in zip(lh_inner_targets, rh_targets):
        #         dmodel = createDoubleSupportActionModel(state,
        #                                                 actuation,
        #                                                 rob_data,
        #                                                 x0,
        #                                                 lf_id,
        #                                                 rf_id,
        #                                                 lh_id,
        #                                                 rh_id,
        #                                                 lh_target=lhand_t,
        #                                                 rh_target=rhand_t)
        #         model_seqs += createSequence([dmodel], DT, N_rhand_to_door)
        #
        # elif i == 3:
        #     DT = 0.03
        #     # Start transferring weight to front foot
        #     N_base_to_ffoot = 15  # knots for left hand reaching
        #     for base_t in base_ffoot_targets:
        #         dmodel = createSingleSupportHandActionModel(state,
        #                                                     actuation,
        #                                                     x0,
        #                                                     lf_id,
        #                                                     rf_id,
        #                                                     lh_id,
        #                                                     rh_id,
        #                                                     base_id,
        #                                                     base_t,
        #                                                     lhand_target=door_l_inner_location,
        #                                                     lh_contact=False,
        #                                                     rhand_target=door_r_inner_location,
        #                                                     rh_contact=False)
        #         model_seqs += createSequence([dmodel], DT, N_base_to_ffoot)
        #
        # elif i == 4:
        #     DT = 0.015
        #     # Using left-hand and right-foot supports, pass right-leg through door
        #     N_base_square_up = 40  # knots per waypoint to pass through door
        #     for base_t, rfoot_t in zip(base_outof_targets, rf_targets):
        #         dmodel = createSingleSupportHandActionModel(state,
        #                                                     actuation,
        #                                                     x0,
        #                                                     lf_id,
        #                                                     rf_id,
        #                                                     lh_id,
        #                                                     rh_id,
        #                                                     base_id,
        #                                                     base_t, rfoot_target=rfoot_t,
        #                                                     lhand_target=lhand_inner_contact_pos,
        #                                                     lh_contact=True,
        #                                                     lh_rpy=[np.pi/2, 0., 0.],
        #                                                     rhand_target=rhand_inner_contact_pos,
        #                                                     rh_contact=True,
        #                                                     rh_rpy=[-np.pi/2, 0., 0.])
        #         model_seqs += createSequence([dmodel], DT, N_base_square_up)
        #
        # elif i == 5:
        #     DT = 0.02
        #     # Push forward while straightening up torso
        #     N_straighten_torso = 20  # knots per waypoint to pass through door
        #     dmodel = createSingleSupportHandActionModel(state,
        #                                                 actuation,
        #                                                 x0,
        #                                                 lf_id,
        #                                                 rf_id,
        #                                                 lh_id,
        #                                                 rh_id,
        #                                                 base_id,
        #                                                 base_outof_targets[-1],
        #                                                 lhand_target=lhand_inner_contact_pos,
        #                                                 lh_contact=True,
        #                                                 rhand_target=rhand_inner_contact_pos,
        #                                                 rh_contact=True,
        #                                                 ang_weights=0.1)
        #     model_seqs += createSequence([dmodel], DT, N_straighten_torso)
        #
        # elif i == 6:
        #     DT = 0.02
        #     # Finish straightening torso
        #     N_straighten_torso = 30  # knots per waypoint to pass through door
        #     dmodel = createSingleSupportHandActionModel(state,
        #                                                 actuation,
        #                                                 x0,
        #                                                 lf_id,
        #                                                 rf_id,
        #                                                 lh_id,
        #                                                 rh_id,
        #                                                 base_id,
        #                                                 base_outof_targets[-1],
        #                                                 ang_weights=0.8)
        #     model_seqs += createSequence([dmodel], DT, N_straighten_torso)

        problem = crocoddyl.ShootingProblem(x0, sum(model_seqs, [])[:-1], model_seqs[-1][-1])
        fddp[i] = crocoddyl.SolverFDDP(problem)

        # Adding callbacks to inspect the evolution of the solver (logs are printed in the terminal)
        fddp[i].setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

        # Solver settings
        max_iter = 150
        fddp[i].th_stop = 1e-3

        # Set initial guess
        xs = [x0] * (fddp[i].problem.T + 1)
        us = fddp[i].problem.quasiStatic([x0] * fddp[i].problem.T)
        print("Problem solved:", fddp[i].solve(xs, us, max_iter))
        print("Number of iterations:", fddp[i].iter)
        print("Total cost:", fddp[i].cost)
        print("Gradient norm:", fddp[i].stoppingCriteria())

        # Set final state as initial state of next phase
        x0 = fddp[i].xs[-1]

    # Creating display
    if B_VISUALIZE:
        save_freq = 1
        display = vis_tools.MeshcatPinocchioAnimation(rob_model, col_model, vis_model,
                          rob_data, vis_data, ctrl_freq=1/DT, save_freq=save_freq)
        # display.add_robot("door", door_model, door_col_model, door_vis_model, door_ini_pos, door_ini_quat)
        display.display_targets("lfoot_target", lf_targets, [1, 0, 0])
        # display.display_targets("rfoot_target", rf_targets, [0, 0, 1])
        display.display_targets("lhand_target", lh_targets, [0.5, 0, 0])
        # display.display_targets("lhand_inner_targets", lh_inner_targets, [0.5, 0, 0])
        # display.display_targets("lhand_ready_targets", lh_ready_targets, [0.5, 0, 0])
        # display.display_targets("rhand_target", rh_targets, [0, 0, 0.5])
        # display.display_targets("rhand_ready_targets", rh_ready_targets, [0, 0, 0.5])
        display.display_targets("base_pass_target", base_into_targets, [0, 1, 0])
        # display.display_targets("base_ffoot_targets", base_ffoot_targets, [0, 1, 0])
        # display.display_targets("base_square_target", base_outof_targets, [0, 1, 0])
        display.add_arrow("forces/l_ankle_ie", color=[1, 0, 0])
        display.add_arrow("forces/r_ankle_ie", color=[0, 0, 1])
        display.add_arrow("forces/l_wrist_pitch", color=[0, 1, 0])
        display.add_arrow("forces/r_wrist_pitch", color=[0, 1, 0])
        # display.displayForcesFromCrocoddylSolver(fddp)
        display.displayFromCrocoddylSolver(fddp)
        viz_to_hide = list(("lfoot_target", "rfoot_target", "lhand_target", "lhand_inner_targets",
                       "rhand_target", "base_pass_target", "base_ffoot_targets", "base_square_target"))
        display.hide_visuals(viz_to_hide)

    fig_idx = 1
    if B_SHOW_JOINT_PLOTS:
        for it in fddp:
            log = it.getCallbacks()[0]
            xs_reduced = np.array(log.xs)[:, [l_constr_ids[0], l_constr_ids[1],
                                              r_constr_ids[0], r_constr_ids[1]]]
            us_reduced = np.array(log.us)[:, [l_constr_ids_u[0], l_constr_ids_u[1],
                                              r_constr_ids_u[0], r_constr_ids_u[1]]]
            crocoddyl.plotOCSolution(xs_reduced, us_reduced, figIndex=fig_idx, show=False)
            fig_idx += 1
            # crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads,
            #                           log.stops, log.steps, figIndex=fig_idx, show=False)
            # fig_idx +=1

    if B_SHOW_GRF_PLOTS:
        # Note: contact_links are l_ankle_ie, r_ankle_ie, l_wrist_pitch, r_wrist_pitch
        sim_steps_list = [len(fddp[i].xs) for i in range(len(fddp))]
        sim_steps = np.sum(sim_steps_list) + 1
        sim_time = np.zeros((sim_steps,))
        rf_lfoot, rf_rfoot, rf_lwrist, rf_rwrist = np.zeros((3, sim_steps)), \
            np.zeros((3, sim_steps)), np.zeros((3, sim_steps)), np.zeros((3, sim_steps))
        time_idx = 0
        for it in fddp:
            rf_list = vis_tools.get_force_trajectory_from_solver(it)
            for rf_t in rf_list:
                for contact in rf_t:
                    # determine contact link
                    cur_link = int(contact['key'])
                    if rob_model.names[cur_link] == "l_ankle_ie":
                        rf_lfoot[:, time_idx] = contact['f'].linear
                    elif rob_model.names[cur_link] == "r_ankle_ie":
                        rf_rfoot[:, time_idx] = contact['f'].linear
                    elif rob_model.names[cur_link] == "l_wrist_pitch":
                        rf_lwrist[:, time_idx] = contact['f'].linear
                    elif rob_model.names[cur_link] == "r_wrist_pitch":
                        rf_rwrist[:, time_idx] = contact['f'].linear
                    else:
                        print("ERROR: Non-specified contact")
                dt = it.problem.runningModels[0].dt     # assumes constant dt over fddp sequence
                sim_time[time_idx+1] = sim_time[time_idx] + dt
                time_idx += 1

        plot_vector_traj(sim_time, rf_lfoot.T, 'RF LFoot (Local)', Fxyz_labels)
        plot_vector_traj(sim_time, rf_rfoot.T, 'RF RFoot (Local)', Fxyz_labels)
        plot_vector_traj(sim_time, rf_lwrist.T, 'RF LWrist (Local)', Fxyz_labels)
        plot_vector_traj(sim_time, rf_rwrist.T, 'RF RWrist (Local)', Fxyz_labels)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=0,
                        help="Contact sequence to solve for")
    args = parser.parse_args()
    main(args)
