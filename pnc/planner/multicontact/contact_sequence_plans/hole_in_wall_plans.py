from pnc.planner.multicontact.kin_feasibility.planner_surface_contact import PlannerSurfaceContact, MotionFrameSequencer
from util.environment_creator import HoleInWallObstructed
import numpy as np

from vision.iris import IrisGeomInterface, IrisRegionsManager


def get_on_lf_balanced_contact_seq(hole: HoleInWallObstructed,
                                starting_pose:  dict[str: np.ndarray],
                                robot_name: str = 'g1',
                                B_USE_KNEES: bool = True):
    # TODO replace hard coded values with values obtained through HoleInWallObstructed
    step_length = 0.44
    side_step = 0.0
    door_l_inner_location = np.array([0.34, 0.32, 1.1])
    door_r_inner_location = np.array([0.34, -0.32, 1.1])
    # door_r_inner_location = np.array([0.34, -0.15, 1.15])
    # door_r_inner_location = np.array([0.34, -0.25, 1.15])
    ft_kn_offset = np.array([0.15, 0., 0.28])
    starting_lh_pos = starting_pose['LH']
    starting_rh_pos = starting_pose['RH']
    starting_torso_pos = starting_pose['torso']
    if B_USE_KNEES:
        final_lkn_pos = starting_pose['L_knee'] + np.array([step_length, side_step, 0.])
        final_rkn_pos = starting_pose['R_knee'] + np.array([step_length, side_step, 0.])
    final_lf_pos = starting_pose['LF'] + np.array([step_length, side_step, 0.])
    final_rf_pos = starting_pose['RF'] + np.array([step_length, side_step, 0.])
    final_torso_pos = starting_pose['torso'] + np.array([step_length, side_step, 0.])
    final_rh_pos = starting_pose['RH'] + np.array([step_length, side_step, 0.])
    final_lh_pos = starting_pose['LH'] + np.array([step_length, side_step, 0.])
    if robot_name == 'g1':
        intermediate_lf_pos = np.array([0.35, (starting_pose['LF'][1] + final_lf_pos[1])/2, 0.44])
    else:
        raise NotImplementedError('Robot name {} not implemented'.format(robot_name))

    # initialize fixed and motion frame sets
    fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

    # ---- Step 1: L and R hand to frame
    if B_USE_KNEES:
        fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
    else:
        fixed_frames.append(['LF', 'RF'])   # frames that must not move
    if robot_name == 'g1':
        motion_frames_seq.add_motion_frame({
                                            'LH': door_l_inner_location,
                                            'RH': door_r_inner_location,
                                            })
    elif robot_name == 'ergoCub':
        motion_frames_seq.add_motion_frame({
                                            'LH': door_l_inner_location,
                                            'RH': door_r_inner_location,
                                            })
    lh_contact_front = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
    # lh_contact_front.set_contact_breaking_velocity(np.array([0, -1, 0.]))
    rh_contact_front = PlannerSurfaceContact('RH', np.array([0, 1, 0]))
    # rh_contact_front = PlannerSurfaceContact('RH', np.array([0., 0.6062, 0.7953]))
    # rh_contact_front = PlannerSurfaceContact('RH', np.array([0., 0.3857,0.9226]))
    # rh_contact_front.set_contact_breaking_velocity(np.array([0, 1, 0.]))
    motion_frames_seq.add_contact_surfaces([lh_contact_front, rh_contact_front])

    # ---- Step 2: step on knee-knocker with left foot
    if robot_name == 'g1':
        if B_USE_KNEES:
            fixed_frames.append(['RF', 'R_knee', 'LH', 'RH'])   # frames that must not move
        else:
            fixed_frames.append(['RF', 'LH', 'RH'])   # frames that must not move
    elif robot_name == 'ergoCub':
        fixed_frames.append(['RF', 'R_knee', 'LH', 'RH'])   # added RH back
    if B_USE_KNEES:
        motion_frames_seq.add_motion_frame({
                            # 'RH': intermediate_rh_pos,      # added for ergoCub
                            'LF': intermediate_lf_pos,
                            'L_knee': intermediate_lf_pos + ft_kn_offset})
    else:
        motion_frames_seq.add_motion_frame({
                            'LF': intermediate_lf_pos,
                            })
    lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([lf_contact_over])

    # ---- Step 3: step through door with right foot
    if robot_name == 'g1':
        if B_USE_KNEES:
            fixed_frames.append(['LF', 'L_knee', 'LH', 'RH'])   # frames that must not move
        else:
            fixed_frames.append(['LF', 'LH', 'RH'])   # frames that must not move
    elif robot_name == 'ergoCub':
        fixed_frames.append(['LF', 'L_knee', 'LH', 'RH'])  # # added RH back
    if B_USE_KNEES:
        motion_frames_seq.add_motion_frame({
                            'R_knee': final_rf_pos + ft_kn_offset,
                            'RF': final_rf_pos})
    else:
        motion_frames_seq.add_motion_frame({
                            'RF': final_rf_pos})
    rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([rf_contact_over])

    # ---- Step 4: balance + return to zero configuration
    if B_USE_KNEES:
        fixed_frames.append(['RF', 'R_knee', 'RH'])
        motion_frames_seq.add_motion_frame({
            'torso': final_torso_pos,
            'LF': final_lf_pos,
            'L_knee': final_lf_pos + ft_kn_offset,
            # 'RH': final_rh_pos,
            'LH': final_lh_pos
        })
    else:
        fixed_frames.append(['RF', 'RH'])
        motion_frames_seq.add_motion_frame({
            'torso': final_torso_pos,
            'LF': final_lf_pos,
            'LH': final_lh_pos
        })
    lf_square_up = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([lf_square_up])

    # ---- Step 5: balance + return to zero configuration
    if B_USE_KNEES:
        fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH'])
    else:
        fixed_frames.append(['torso', 'LF', 'RF', 'LH'])
    motion_frames_seq.add_motion_frame({'RH': final_rh_pos})

    return fixed_frames, motion_frames_seq

def get_on_rf_balanced_contact_seq(hole: HoleInWallObstructed,
                                starting_pose:  dict[str: np.ndarray],
                                robot_name: str = 'g1',
                                B_USE_KNEES: bool = True):
    # TODO replace hard coded values with values obtained through HoleInWallObstructed
    step_length = 0.46
    side_step = 0.0
    door_l_inner_location = np.array([0.34, 0.32, 1.0])
    door_r_inner_location = np.array([0.34, -0.3, 1.0])
    # door_r_inner_location = np.array([0.34, -0.15, 1.15])
    # door_r_inner_location = np.array([0.34, -0.28, 1.])
    ft_kn_offset = np.array([0.15, 0., 0.28])
    starting_lh_pos = starting_pose['LH']
    starting_rh_pos = starting_pose['RH']
    starting_torso_pos = starting_pose['torso']
    if B_USE_KNEES:
        final_lkn_pos = starting_pose['L_knee'] + np.array([step_length, side_step, 0.])
        final_rkn_pos = starting_pose['R_knee'] + np.array([step_length, side_step, 0.])
    final_lf_pos = starting_pose['LF'] + np.array([step_length, side_step, 0.])
    final_rf_pos = starting_pose['RF'] + np.array([step_length, side_step, 0.])
    final_torso_pos = starting_pose['torso'] + np.array([step_length + 0.04, side_step, 0.])
    final_rh_pos = starting_pose['RH'] + np.array([step_length, side_step, 0.])
    final_lh_pos = starting_pose['LH'] + np.array([step_length, side_step, 0.])
    if robot_name == 'g1':
        intermediate_rf_pos = np.array([0.35, (starting_pose['RF'][1] + final_rf_pos[1])/2, 0.44])
    else:
        raise NotImplementedError('Robot name {} not implemented'.format(robot_name))

    # initialize fixed and motion frame sets
    fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

    # ---- Step 1: L and R hand to frame
    if B_USE_KNEES:
        fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
    else:
        fixed_frames.append(['LF', 'RF'])   # frames that must not move
    if robot_name == 'g1':
        motion_frames_seq.add_motion_frame({
                                            'LH': door_l_inner_location,
                                            'RH': door_r_inner_location,
                                            })
    elif robot_name == 'ergoCub':
        motion_frames_seq.add_motion_frame({
                                            'LH': door_l_inner_location,
                                            'RH': door_r_inner_location,
                                            })
    lh_contact_front = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
    # lh_contact_front.set_contact_breaking_velocity(np.array([0, -1, 0.]))
    # rh_contact_front = PlannerSurfaceContact('RH', np.array([0, 1, 0]))
    # rh_contact_front = PlannerSurfaceContact('RH', np.array([0., 0.6062, 0.7953]))
    rh_contact_front = PlannerSurfaceContact('RH', np.array([0., 0.3857,0.9226]))
    # rh_contact_front.set_contact_breaking_velocity(np.array([0, 1, 0.]))
    motion_frames_seq.add_contact_surfaces([lh_contact_front, rh_contact_front])

    # ---- Step 2: step on knee-knocker with right foot
    if robot_name == 'g1':
        if B_USE_KNEES:
            fixed_frames.append(['LF', 'L_knee', 'LH', 'RH'])   # frames that must not move
        else:
            fixed_frames.append(['LF', 'LH', 'RH'])   # frames that must not move
    elif robot_name == 'ergoCub':
        fixed_frames.append(['LF', 'L_knee', 'LH', 'RH'])   # added RH back
    if B_USE_KNEES:
        motion_frames_seq.add_motion_frame({
                            # 'RH': intermediate_rh_pos,      # added for ergoCub
                            'RF': intermediate_rf_pos,
                            'R_knee': intermediate_rf_pos + ft_kn_offset})
    else:
        motion_frames_seq.add_motion_frame({
                            'RF': intermediate_rf_pos,
                            })
    rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([rf_contact_over])

    # ---- Step 3: step through door with right foot
    if robot_name == 'g1':
        if B_USE_KNEES:
            fixed_frames.append(['RF', 'R_knee', 'LH', 'RH'])   # frames that must not move
        else:
            fixed_frames.append(['RF', 'LH', 'RH'])   # frames that must not move
    elif robot_name == 'ergoCub':
        fixed_frames.append(['RF', 'R_knee', 'LH', 'RH'])  # # added RH back
    if B_USE_KNEES:
        motion_frames_seq.add_motion_frame({
                            'L_knee': final_lf_pos + ft_kn_offset,
                            'LF': final_lf_pos})
    else:
        motion_frames_seq.add_motion_frame({
                            'LF': final_lf_pos})
    lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([lf_contact_over])

    # ---- Step 4: balance + return to zero configuration
    if B_USE_KNEES:
        fixed_frames.append(['LF', 'L_knee', 'RH'])
        motion_frames_seq.add_motion_frame({
            'torso': final_torso_pos,
            'RF': final_rf_pos,
            'R_knee': final_rf_pos + ft_kn_offset,
            # 'RH': final_rh_pos,
            'LH': final_lh_pos
        })
    else:
        fixed_frames.append(['LF', 'RH'])
        motion_frames_seq.add_motion_frame({
            'torso': final_torso_pos,
            'RF': final_rf_pos,
            'LH': final_lh_pos
        })
    rf_square_up = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([rf_square_up])

    # ---- Step 5: balance + return to zero configuration
    if B_USE_KNEES:
        fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH'])
    else:
        fixed_frames.append(['torso', 'LF', 'RF', 'LH'])
    motion_frames_seq.add_motion_frame({'RH': final_rh_pos})

    return fixed_frames, motion_frames_seq

def compute_iris_regions_mgr(hole: HoleInWallObstructed,
                             starting_pose: dict[str, np.ndarray],
                             motion_frames_seq: MotionFrameSequencer,
                             b_use_knees: bool = True):
    # load obstacle, domain, and start / end seed for IRIS
    obstacles = hole.obstacles
    domain = hole.domain
    # shift (feet) iris seed to get nicer IRIS region
    step_length = 0.46
    side_step = 0.0
    iris_lf_shift = np.array([0.2, 0., 0.])
    iris_rf_shift = np.array([0.2, 0., 0.])
    iris_kn_shift = np.array([0.0, 0., 0.0])
    iris_rh_shift = np.array([-0.2, 0., 0.0])

    starting_torso_pos = starting_pose['torso']
    starting_lf_pos = starting_pose['LF']
    starting_lh_pos = starting_pose['LH']
    starting_rf_pos = starting_pose['RF']
    starting_rh_pos = starting_pose['RH']
    if b_use_knees:
        starting_lkn_pos = starting_pose['L_knee']
        starting_rkn_pos = starting_pose['R_knee']
        final_lkn_pos = starting_pose['L_knee'] + np.array([step_length, side_step, 0.])
        final_rkn_pos = starting_pose['R_knee'] + np.array([step_length, side_step, 0.])

    final_lf_pos = starting_pose['LF'] + np.array([step_length, side_step, 0.])
    final_rf_pos = starting_pose['RF'] + np.array([step_length, side_step, 0.])
    final_torso_pos = starting_pose['torso'] + np.array([step_length, side_step, 0.])
    final_rh_pos = starting_pose['RH'] + np.array([step_length, side_step, 0.])
    final_lh_pos = starting_pose['LH'] + np.array([step_length, side_step, 0.])


    # create dictionary of safe regions
    safe_torso_start_region = IrisGeomInterface(obstacles, domain, starting_torso_pos)
    safe_lf_start_region = IrisGeomInterface(obstacles, domain, starting_lf_pos + iris_lf_shift)
    safe_lh_start_region = IrisGeomInterface(obstacles, domain, starting_lh_pos)
    safe_rf_start_region = IrisGeomInterface(obstacles, domain, starting_rf_pos + iris_rf_shift)
    safe_rh_start_region = IrisGeomInterface(obstacles, domain, starting_rh_pos + iris_rh_shift)
    safe_torso_final_region = IrisGeomInterface(obstacles, domain, final_torso_pos + np.array([-0.1, 0., 0.]))
    safe_lf_final_region = IrisGeomInterface(obstacles, domain, final_lf_pos)
    safe_lh_final_region = IrisGeomInterface(obstacles, domain, final_lh_pos)
    safe_rf_final_region = IrisGeomInterface(obstacles, domain, final_rf_pos)
    safe_rh_final_region = IrisGeomInterface(obstacles, domain, final_rh_pos)
    safe_regions_mgr_dict = {'torso': IrisRegionsManager(safe_torso_start_region, safe_torso_final_region),
                             'LF': IrisRegionsManager(safe_lf_start_region, safe_lf_final_region),
                             'LH': IrisRegionsManager(safe_lh_start_region, safe_lh_final_region),
                             'RF': IrisRegionsManager(safe_rf_start_region, safe_rf_final_region),
                             'RH': IrisRegionsManager(safe_rh_start_region, safe_rh_final_region)}
    if b_use_knees:
        safe_lk_start_region = IrisGeomInterface(obstacles, domain, starting_lkn_pos + np.array([0.1, 0., -0.05]))
        safe_rk_start_region = IrisGeomInterface(obstacles, domain, starting_rkn_pos + np.array([0.1, 0., -0.05]))
        safe_lk_final_region = IrisGeomInterface(obstacles, domain, final_lkn_pos + np.array([-0.18, 0., -0.15]))
        safe_rk_final_region = IrisGeomInterface(obstacles, domain, final_rkn_pos + np.array([-0.18, 0., -0.15]))

        safe_regions_mgr_dict['L_knee'] = IrisRegionsManager(safe_lk_start_region, safe_lk_final_region)
        safe_regions_mgr_dict['R_knee'] = IrisRegionsManager(safe_rk_start_region, safe_rk_final_region)

    # loop through each of the planned steps to make sure we have an IRIS regions for each
    # Note: we remove the IRIS regions where we have bad overlap
    for fr_dict in motion_frames_seq.motion_frame_lst:
        for fr_name, pos in fr_dict.items():
            iris_mgr = safe_regions_mgr_dict[fr_name]

            b_skip = False
            # create new Iris Region if it isn't already present in iris_list
            overlap_thresh = 0.2
            for ir_seed in iris_mgr.iris_list:
                # skip creating IRIS regions for existing seed
                if np.linalg.norm(pos - ir_seed.seed_pos) < overlap_thresh:
                    b_skip = True
                    break

            if b_skip or fr_name == 'R_knee' or fr_name == 'L_knee':
                continue

            # add the new IRIS seed
            next_ir = IrisGeomInterface(obstacles, domain, pos)
            iris_mgr.addIris([next_ir])

    # compute and connect IRIS from start to goal
    for _, irm in safe_regions_mgr_dict.items():
        irm.computeIris()
        hint_seed = hole.obstacles[1].ChebyshevCenter()
        hint_seed[2] = 0.7
        irm.connectIrisListSeeds("centroid", hint_seed)

    return safe_regions_mgr_dict