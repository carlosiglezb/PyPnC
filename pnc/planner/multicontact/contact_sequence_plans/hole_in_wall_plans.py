from pnc.planner.multicontact.kin_feasibility.planner_surface_contact import PlannerSurfaceContact, MotionFrameSequencer
from pnc.planner.multicontact.kin_feasibility.environment_inflator import (
    EnvironmentInflator, collect_contact_anchors)
from util.environment_creator import HoleInWallObstructed
import numpy as np

from vision.iris import IrisGeomInterface, IrisRegionsManager


def _clear_seed_of_collisions(seed: np.ndarray, obstacles,
                              push_dir: np.ndarray = np.array([1., 0., 0.]),
                              step: float = 0.02, max_iters: int = 100) -> np.ndarray:
    """Nudge `seed` along push_dir in small increments until it lies outside
    every obstacle in `obstacles`.

    The hand-tuned IRIS seed offsets below (e.g. pulling a foot's "final"
    seed back by a fixed x amount) assume a step displacement that doesn't
    always hold -- e.g. for the on-knocker sequence this used to place the
    LF/RF final seed squarely inside the knee-knocker base. IRIS silently
    accepts an in-collision seed (it does not raise) and returns a bogus
    region that isn't real obstacle-free space, which can make two otherwise
    disconnected regions appear connected. Nudging the seed clear up front
    prevents that failure mode.
    """
    pos = np.array(seed, dtype=float).copy()
    for _ in range(max_iters):
        if not any(obs.PointInSet(pos) for obs in obstacles):
            return pos
        pos = pos + step * push_dir
    return pos


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
                             b_use_knees: bool = True,
                             sca_robot_geometry=None):
    # load obstacle, domain, and start / end seed for IRIS
    obstacles = hole.obstacles
    domain = hole.domain

    # per-frame obstacle sets, inflated by each frame's collision-sphere radius
    # with contact-face exemptions (swept-sphere collision avoidance); frames
    # without a sphere primitive (e.g. torso capsule) keep the shared obstacles
    per_frame_obstacles = None
    if sca_robot_geometry is not None:
        inflator = EnvironmentInflator(obstacles, sca_robot_geometry)
        anchors = collect_contact_anchors(starting_pose, motion_frames_seq)
        per_frame_obstacles = inflator.per_frame_obstacles(anchors)

    def obs_for(fr):
        if per_frame_obstacles is not None and fr in per_frame_obstacles:
            return per_frame_obstacles[fr]
        return obstacles
    # shift (feet) iris seed to get nicer IRIS region
    step_length = 0.46
    side_step = 0.0
    # Note: the feet start ~5 cm ahead of the base with the asymmetric-pitch
    # initial stance, so a +0.15 shift places the seed at the same world x
    # (just before the knee-knocker front face at x=0.26) as +0.2 did with the
    # old symmetric stance.
    iris_lf_shift = np.array([0.15, 0., 0.])
    iris_rf_shift = np.array([0.15, 0., 0.])
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


    # create dictionary of safe regions (seeds are nudged in +x if the hand-tuned
    # offset above happens to land them inside an obstacle -- see
    # _clear_seed_of_collisions)
    safe_torso_start_region = IrisGeomInterface(obs_for('torso'), domain,
        _clear_seed_of_collisions(starting_torso_pos, obs_for('torso')))
    safe_lf_start_region = IrisGeomInterface(obs_for('LF'), domain,
        _clear_seed_of_collisions(starting_lf_pos + iris_lf_shift, obs_for('LF')))
    safe_lh_start_region = IrisGeomInterface(obs_for('LH'), domain,
        _clear_seed_of_collisions(starting_lh_pos, obs_for('LH')))
    safe_rf_start_region = IrisGeomInterface(obs_for('RF'), domain,
        _clear_seed_of_collisions(starting_rf_pos + iris_rf_shift, obs_for('RF')))
    safe_rh_start_region = IrisGeomInterface(obs_for('RH'), domain,
        _clear_seed_of_collisions(starting_rh_pos + iris_rh_shift, obs_for('RH')))
    safe_torso_final_region = IrisGeomInterface(obs_for('torso'), domain,
        _clear_seed_of_collisions(final_torso_pos + np.array([-0.1, 0., 0.]), obs_for('torso')))
    safe_lf_final_region = IrisGeomInterface(obs_for('LF'), domain,
        _clear_seed_of_collisions(final_lf_pos + np.array([-0.1, 0., 0.]), obs_for('LF')))
    safe_lh_final_region = IrisGeomInterface(obs_for('LH'), domain,
        _clear_seed_of_collisions(final_lh_pos, obs_for('LH')))
    safe_rf_final_region = IrisGeomInterface(obs_for('RF'), domain,
        _clear_seed_of_collisions(final_rf_pos + np.array([-0.1, 0., 0.]), obs_for('RF')))
    safe_rh_final_region = IrisGeomInterface(obs_for('RH'), domain,
        _clear_seed_of_collisions(final_rh_pos, obs_for('RH')))
    safe_regions_mgr_dict = {'torso': IrisRegionsManager(safe_torso_start_region, safe_torso_final_region),
                             'LF': IrisRegionsManager(safe_lf_start_region, safe_lf_final_region),
                             'LH': IrisRegionsManager(safe_lh_start_region, safe_lh_final_region),
                             'RF': IrisRegionsManager(safe_rf_start_region, safe_rf_final_region),
                             'RH': IrisRegionsManager(safe_rh_start_region, safe_rh_final_region)}
    if b_use_knees:
        # knees also start ~8 cm further forward in the asymmetric stance; the
        # smaller +0.02 x-shift keeps the seed out of the knee-knocker box
        # (x in [0.26, 0.34], z < 0.4) at the same world x as the old +0.1
        safe_lk_start_region = IrisGeomInterface(obs_for('L_knee'), domain,
            _clear_seed_of_collisions(starting_lkn_pos + np.array([0.02, 0., -0.05]), obs_for('L_knee')))
        safe_rk_start_region = IrisGeomInterface(obs_for('R_knee'), domain,
            _clear_seed_of_collisions(starting_rkn_pos + np.array([0.02, 0., -0.05]), obs_for('R_knee')))
        safe_lk_final_region = IrisGeomInterface(obs_for('L_knee'), domain,
            _clear_seed_of_collisions(final_lkn_pos + np.array([-0.18, 0., -0.15]), obs_for('L_knee')))
        safe_rk_final_region = IrisGeomInterface(obs_for('R_knee'), domain,
            _clear_seed_of_collisions(final_rkn_pos + np.array([-0.18, 0., -0.15]), obs_for('R_knee')))

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
            next_ir = IrisGeomInterface(obs_for(fr_name), domain, pos)
            iris_mgr.addIris([next_ir])

    # compute and connect IRIS from start to goal
    for fr_name, irm in safe_regions_mgr_dict.items():
        irm.computeIris()
        hint_seed = hole.obstacles[1].ChebyshevCenter()
        hint_seed[2] = 0.7
        irm.connectIrisListSeeds("centroid", hint_seed, label=fr_name)

    return safe_regions_mgr_dict