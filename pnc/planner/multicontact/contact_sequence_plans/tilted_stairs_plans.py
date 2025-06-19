from pnc.planner.multicontact.kin_feasibility.planner_surface_contact import PlannerSurfaceContact, MotionFrameSequencer
from util.environment_creator import TiltedStairs
import numpy as np

from vision.iris import IrisGeomInterface, IrisRegionsManager


# ---------------------------
#   Tilted Stairs Settings
# ---------------------------
def get_opposing_limbs_contact_sequence(stairs: TiltedStairs,
                                        starting_pose: dict[str: np.ndarray],
                                        robot_name: str = 'g1',
                                        b_use_knees: bool = True):
    box_width = stairs.box_width
    box_depth = stairs.box_depth
    box_h1_left = stairs.box_h1_left
    box_h2_left = stairs.box_h2_left
    box_h1_right = stairs.box_h1_right
    box_h2_right = stairs.box_h2_right
    if robot_name == 'g1':
        # G1 settings
        ankle_height = 0.08
        torso_hand_height = 0.08
        rh1_height = 1.0
        lh2_height = 1.4
        rh3_height = 1.7
    elif robot_name == 'ergoCub':
        # ErgoCub settings
        ankle_height = 0.1
        torso_hand_height = -0.05
        rh1_height = 1.1
        lh2_height = 1.6
        rh3_height = 1.9
    else:
        raise ValueError(f"Unknown tilted stair settings for robot: {robot_name}")
    delta_h_left = (box_h2_left - box_h1_left)
    delta_h_right = (box_h2_right - box_h1_right)
    left_step_normal = np.array([0, -delta_h_left, box_width])
    right_step_normal = np.array([0, delta_h_right, box_width])
    lh_wall_normal = np.array([0, -1, 0])
    rh_wall_normal = np.array([0, 1, 0])
    foot_final_step_normal = np.array([0, 0, 1])

    # get end effector positions via fwd kin
    starting_torso_pos = starting_pose['torso']
    starting_lf_pos = starting_pose['LF']
    starting_lh_pos = starting_pose['LH']
    starting_rf_pos = starting_pose['RF']
    starting_rh_pos = starting_pose['RH']
    if b_use_knees:
        starting_lkn_pos = starting_pose['L_knee']
        starting_rkn_pos = starting_pose['R_knee']

    final_lf_pos = np.array([0.2 + 2.5 * box_depth , 0.1, 1. + ankle_height])
    final_rf_pos = np.array([0.2 + 2.5 * box_depth , -0.1, 1. + ankle_height])
    final_torso_pos = (final_lf_pos + final_rf_pos) / 2 + np.array([0., 0., starting_torso_pos[2]])
    final_rh_pos = final_torso_pos + np.array([0.3, -0.2, torso_hand_height])
    final_lh_pos = final_torso_pos + np.array([0.3, 0.2, torso_hand_height])
    if b_use_knees:
        rough_knee_pos = (starting_lkn_pos - starting_lf_pos)
        scaled_knee_pos = final_lf_pos + rough_knee_pos * 0.3139 / np.linalg.norm(rough_knee_pos)
        final_lkn_pos = scaled_knee_pos
        rough_knee_pos = (starting_rkn_pos - starting_rf_pos)
        scaled_knee_pos = final_rf_pos + rough_knee_pos * 0.3139 / np.linalg.norm(rough_knee_pos)
        final_rkn_pos = scaled_knee_pos

    # intermediate locations
    rh1_wall = np.array([0.34, -0.32, rh1_height])
    lf_step1 = np.array([0.35, box_width/2, (box_h1_left + box_h2_left)/2 + ankle_height])
    lh_wall_step_12 = np.array([0.2 + box_depth, box_width - 0.03, lh2_height])
    rh_wall_final_step = np.array([0.25 + box_depth, -(box_width - 0.03), rh3_height])
    rf_step2 = np.array([0.32+ box_depth, -box_width/2, (box_h1_right + box_h2_right)/2 + ankle_height])

    # initialize fixed and motion frame sets
    fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

    # ---- Step 1: R hand to frame
    if b_use_knees:
        fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])  # frames that must not move
    else:
        fixed_frames.append(['LF', 'RF'])  # frames that must not move
    motion_frames_seq.add_motion_frame({
        'RH': rh1_wall,
    })
    rh_wall1_contact = PlannerSurfaceContact('RH', rh_wall_normal)
    # rh_wall1_contact.set_contact_breaking_velocity(rh_wall_normal)
    motion_frames_seq.add_contact_surfaces([rh_wall1_contact])

    # ---- Step 2: step on left tilted step (and RH wall)
    if b_use_knees:
        rough_knee_pos = np.array([0.08, 0., 0.25])
        scaled_knee_pos = lf_step1 + rough_knee_pos * 0.3139 / np.linalg.norm(rough_knee_pos)
        fixed_frames.append(['RF', 'R_knee', 'RH'])  # frames that must not move
        motion_frames_seq.add_motion_frame({
            'LF': lf_step1,
            'L_knee': scaled_knee_pos})
    else:
        fixed_frames.append(['RF', 'RH'])  # frames that must not move
        motion_frames_seq.add_motion_frame({
            'LF': lf_step1})
    lf_step1_contact = PlannerSurfaceContact('LF', left_step_normal)
    # lf_step1_contact.set_contact_breaking_velocity(foot_final_step_normal)
    motion_frames_seq.add_contact_surfaces([lf_step1_contact])

    # ---- Step 3: move to second step with RF
    if b_use_knees:
        rough_knee_pos = np.array([0.08, 0., 0.25])
        scaled_knee_pos = rf_step2 + rough_knee_pos * 0.3139 / np.linalg.norm(rough_knee_pos)
        fixed_frames.append(['LF', 'L_knee', 'RH'])  # frames that must not move
        motion_frames_seq.add_motion_frame({
            'LH': lh_wall_step_12,
            'R_knee': scaled_knee_pos,
            'RF': rf_step2})
    else:
        fixed_frames.append(['LF', 'RH'])  # frames that must not move
        motion_frames_seq.add_motion_frame({
            'LH': lh_wall_step_12,
            'RF': rf_step2})
    rf_step2_contact = PlannerSurfaceContact('RF', right_step_normal)
    lh_step2_contact = PlannerSurfaceContact('LH', lh_wall_normal)
    motion_frames_seq.add_contact_surfaces([rf_step2_contact, lh_step2_contact])

    # ---- Step 4: step on middle box with LF
    if b_use_knees:
        fixed_frames.append(['RF', 'R_knee', 'LH'])
        motion_frames_seq.add_motion_frame({
            'LF': final_lf_pos,
            'L_knee': final_lkn_pos,
            'RH': rh_wall_final_step,
        })
    else:
        fixed_frames.append(['RF', 'LH'])
        motion_frames_seq.add_motion_frame({
            'LF': final_lf_pos,
        })
    lf_step3_contact = PlannerSurfaceContact('LF', foot_final_step_normal)
    rh_step3_contact = PlannerSurfaceContact('RH', rh_wall_normal)
    motion_frames_seq.add_contact_surfaces([lf_step3_contact, rh_step3_contact])

    # ---- Step 5: step on middle box with RF
    if b_use_knees:
        fixed_frames.append(['LF', 'L_knee', 'RH'])
        motion_frames_seq.add_motion_frame({
            'torso': final_torso_pos,
            'RF': final_rf_pos,
            'R_knee': final_rkn_pos,
            'LH': final_lh_pos,
        })
    else:
        fixed_frames.append(['LF', 'RH'])
        motion_frames_seq.add_motion_frame({
            'torso': final_torso_pos,
            'RF': final_rf_pos,
            'LH': final_lh_pos,
        })
    rf_step4_contact = PlannerSurfaceContact('RF', foot_final_step_normal)
    motion_frames_seq.add_contact_surfaces([rf_step4_contact])

    # ---- Step 6: balance
    if b_use_knees:
        fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH'])
    else:
        fixed_frames.append(['torso', 'LF', 'RF', 'LH'])
    motion_frames_seq.add_motion_frame({'RH': final_rh_pos,})

    return fixed_frames, motion_frames_seq

def compute_stairs_iris_regions_mgr(stairs: TiltedStairs,
                                    starting_pose: dict[str, np.ndarray],
                                    motion_frames_seq: MotionFrameSequencer,
                                    b_use_knees: bool = True):
    # load obstacle, domain, and start / end seed for IRIS
    obstacles = stairs.obstacles
    domain = stairs.domain
    # shift (feet) iris seed to get nicer IRIS region
    iris_lf_shift = np.array([0.0, 0., 0.])
    iris_rf_shift = np.array([0.0, 0., 0.])
    iris_kn_shift = np.array([0.0, 0., 0.0])

    starting_torso_pos = starting_pose['torso']
    starting_lf_pos = starting_pose['LF']
    starting_lh_pos = starting_pose['LH']
    starting_rf_pos = starting_pose['RF']
    starting_rh_pos = starting_pose['RH']
    if b_use_knees:
        starting_lkn_pos = starting_pose['L_knee']
        starting_rkn_pos = starting_pose['R_knee']

    # create dictionary of safe regions
    safe_torso_start_region = IrisGeomInterface(obstacles, domain, starting_torso_pos)
    safe_lf_start_region = IrisGeomInterface(obstacles, domain, starting_lf_pos + iris_lf_shift)
    safe_lh_start_region = IrisGeomInterface(obstacles, domain, starting_lh_pos)
    safe_rf_start_region = IrisGeomInterface(obstacles, domain, starting_rf_pos + iris_rf_shift)
    safe_rh_start_region = IrisGeomInterface(obstacles, domain, starting_rh_pos)
    safe_regions_mgr_dict = {'torso': IrisRegionsManager(safe_torso_start_region),
                             'LF': IrisRegionsManager(safe_lf_start_region),
                             'LH': IrisRegionsManager(safe_lh_start_region),
                             'RF': IrisRegionsManager(safe_rf_start_region),
                             'RH': IrisRegionsManager(safe_rh_start_region)}
    if b_use_knees:
        safe_lk_start_region = IrisGeomInterface(obstacles, domain, starting_lkn_pos + np.array([0.02, 0., -0.05]))
        safe_rk_start_region = IrisGeomInterface(obstacles, domain, starting_rkn_pos)

        safe_regions_mgr_dict['L_knee'] = IrisRegionsManager(safe_lk_start_region)
        safe_regions_mgr_dict['R_knee'] = IrisRegionsManager(safe_rk_start_region)

    # loop through each of the planned steps to make sure we have an IRIS regions for each
    for fr_dict in motion_frames_seq.motion_frame_lst:
        for fr_name, pos in fr_dict.items():
            next_ir = IrisGeomInterface(obstacles, domain, pos)
            safe_regions_mgr_dict[fr_name].addIris([next_ir])

    # compute and connect IRIS from start to goal
    for _, irm in safe_regions_mgr_dict.items():
        irm.computeIris()
        irm.connectIrisListSeeds("volume")

    return safe_regions_mgr_dict
