import numpy as np
import crocoddyl
import pinocchio
import pinocchio as pin

import util.liegroup
from config.multicontact.baseline_planner_config import BaselinePlannerConfig
from config.multicontact.planner_config import PlannerConfig
from pnc.planner.multicontact.crocoddyl_extensions.ActivationModelDistanceQuad import ActivationModelDistanceQuad
from pnc.planner.multicontact.crocoddyl_extensions.ControlBounds import ControlBounds
from pnc.planner.multicontact.crocoddyl_extensions.ResidualDistanceCollision import ResidualDistanceCollision
from pnc.planner.multicontact.dyn_feasibility.HumanoidMulticontactPlanner import ContactSequence
from util.util import so3_from_vec_to_vec

Z_UP =  np.array([0., 0., 1])
mu = 0.9


def get_limb_joint_idx(limb_joint_names: list[str],
                  robot_model: pinocchio.Model):
    # do not count the universe and root joints
    return [robot_model.getJointId(joint_name) - 2 for joint_name in limb_joint_names]

def createMultiFrameActionModel(state: crocoddyl.StateMultibody,
                                actuation: crocoddyl.ActuationModelFloatingBase,
                                x0: np.array,
                                plan_to_model_ids: dict[str, int],
                                frames_in_contact: dict[str: np.array],
                                next_frames_in_contact: dict[str: np.array],
                                frame_targets_dict: dict[str, np.array],
                                joint_names_dict: dict[str, list[str]] = None,
                                planner_weights: PlannerConfig | BaselinePlannerConfig = None,
                                zero_config: np.array = None,
                                v_ref: np.array = None,
                                terminal_step: bool = False,
                                geom_model: pinocchio.GeometryModel = None,
                                robot_model: pinocchio.Model = None,
                                b_sqp: bool = False,
                                b_sca: bool = False):
    desired_config = np.copy(x0)

    # Define the cost sum (cost manager)
    costs = crocoddyl.CostModelSum(state, actuation.nu)

    # Define contacts (e.g., feet / hand supports)
    contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)

    # scale cost of joint velocities to be higher on limbs in contact
    wx_scale =  np.ones(actuation.nu)

    # create contact models for each frame in contact
    for fr_name, fr_plane in frames_in_contact.items():
        fr_id = plan_to_model_ids[fr_name]

        # set the corresponding rotation for all frames in contact
        SE3_ee = pin.SE3.Identity()
        # SE3_ee.rotation = so3_from_vec_to_vec(Z_UP, fr_plane)
        r,p = util.util.vec_to_roll_pitch(fr_plane)
        SE3_ee.rotation = util.util.euler_to_rot([r,p,0])

        if 'H' in fr_name:
            fr_contact = crocoddyl.ContactModel3D(
                state,
                fr_id,
                np.zeros(3),
                pin.LOCAL_WORLD_ALIGNED,
                actuation.nu,
                np.array(planner_weights.WBP_BAUMGARTE_GAINS3D),
            )
        else:
            fr_contact = crocoddyl.ContactModel6D(
                state,
                fr_id,
                SE3_ee,
                pin.LOCAL_WORLD_ALIGNED,
                actuation.nu,
                np.array(planner_weights.WBP_BAUMGARTE_GAINS6D),
            )
        contacts.addContact(fr_name + "_contact", fr_contact)

        # Add friction cone penalization according to foot or hand contact
        floor_rotation = np.eye(3)
        if 'H' in fr_name:
            # surf_cone = crocoddyl.FrictionCone(floor_rotation, mu, 4, True)     # better if False?
            surf_cone = crocoddyl.FrictionCone(SE3_ee.rotation, mu, 4, True)     # better if False?
        else:
            foot_size = planner_weights.FOOT_SIZE
            surf_cone = crocoddyl.WrenchCone(SE3_ee.rotation, mu, np.array(foot_size), 4, True)     # better if False?
            # surf_cone = crocoddyl.WrenchCone(floor_rotation, mu, np.array(foot_size), 4, True)     # better if False?

        # friction cone activation function
        surf_activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(surf_cone.lb, surf_cone.ub)
        )
        if 'H' in fr_name:
            fr_friction = crocoddyl.CostModelResidual(
                state,
                surf_activation_friction,
                crocoddyl.ResidualModelContactFrictionCone(state, fr_id, surf_cone, actuation.nu),
            )
        else:
            fr_friction = crocoddyl.CostModelResidual(
                state,
                surf_activation_friction,
                crocoddyl.ResidualModelContactWrenchCone(state, fr_id, surf_cone, actuation.nu),
            )
        costs.addCost(fr_name + "_friction",
                      fr_friction,
                      planner_weights.WBC_COST_WEIGHTS['friction'])

        # increase cost of joint velocities on limbs in contact
        if joint_names_dict is not None:
            if 'LF' in fr_name:
                jnt_in_contact = get_limb_joint_idx(joint_names_dict['left_leg'], robot_model)
            elif 'RF' in fr_name:
                jnt_in_contact = get_limb_joint_idx(joint_names_dict['right_leg'], robot_model)
            elif 'LH' in fr_name:
                jnt_in_contact = get_limb_joint_idx(joint_names_dict['left_arm'], robot_model)
            elif 'RH' in fr_name:
                jnt_in_contact = get_limb_joint_idx(joint_names_dict['right_arm'], robot_model)
            wx_scale[jnt_in_contact] *= planner_weights.WBP_CONTACT_JVEL_SCALE

    # Add frame-placement cost
    for fr_name, fr_id in plan_to_model_ids.items():
        # skip if no target is provided for this frame (used mostly for baseline planner)
        if fr_name not in frame_targets_dict.keys():
            continue

        # set higher tracking cost on feet
        if 'F' in fr_name:
            w_fr = planner_weights.WBC_FRAME_TRACKING_GAINS['feet']
        elif 'RH' in fr_name:
            w_fr = planner_weights.WBC_FRAME_TRACKING_GAINS['RH']
        elif 'LH' in fr_name:
            w_fr = planner_weights.WBC_FRAME_TRACKING_GAINS['LH']
        elif 'R_knee' in fr_name:
            w_fr = planner_weights.WBC_FRAME_TRACKING_GAINS['R_knee']
        elif 'L_knee' in fr_name:
            w_fr = planner_weights.WBC_FRAME_TRACKING_GAINS['L_knee']
        elif 'torso' in fr_name:
            w_fr = planner_weights.WBC_FRAME_TRACKING_GAINS['torso']
            # if zero_config is not None:
            #     w_fr = np.array([0.1] * 3 + [0.01] * 3)
        else:
            raise ValueError(f"Weights to track frame {fr_name} were not set")

        # set the desired frame position (and orientation for upcoming feet contact planes)
        fr_Mref = pin.SE3.Identity()
        if fr_name in next_frames_in_contact.keys() and 'H' not in fr_name:
            # fr_Mref.rotation = so3_from_vec_to_vec(Z_UP, next_frames_in_contact[fr_name])
            r, p = util.util.vec_to_roll_pitch(next_frames_in_contact[fr_name])
            fr_Mref.rotation = util.util.euler_to_rot([r, p, 0])
        elif fr_name in frames_in_contact.keys() and 'H' not in fr_name:
            # fr_Mref.rotation = so3_from_vec_to_vec(Z_UP, frames_in_contact[fr_name])
            r, p = util.util.vec_to_roll_pitch(frames_in_contact[fr_name])
            fr_Mref.rotation = util.util.euler_to_rot([r, p, 0])
        fr_Mref.translation = frame_targets_dict[fr_name]

        # add as cost
        activation_fr = crocoddyl.ActivationModelWeightedQuad(w_fr ** 2)
        fr_cost = crocoddyl.CostModelResidual(
            state,
            activation_fr,
            crocoddyl.ResidualModelFramePlacement(state, fr_id, fr_Mref, actuation.nu),
        )
        costs.addCost(fr_name + "_goal",
                      fr_cost,
                      planner_weights.WBC_COST_WEIGHTS['frame_goal'])

    #
    # Adding state and control regularization terms
    #
    ulim = state.pinocchio.effortLimit[-(state.nv - 6):]
    weight_by_ulim = ulim
    weight_by_mass = [i.mass for i in state.pinocchio.inertias.tolist()[-(state.nv - 6):]]
    w_x = np.copy(planner_weights.WBC_WEIGHTED_COSTS['xReg'])
    w_x[-actuation.nu:] *= wx_scale     # penalize more jvel of contact limb
    w_x[6:state.nq-1] *= wx_scale    # penalize more jpos of contact limb

    # Change reference state if zero_config is provided, otherwise use the initial state
    if zero_config is not None and terminal_step:
        desired_config[3:state.nq] = zero_config[3:]
        w_x[:3] = 0
    if v_ref is not None:
        desired_config[-state.nv:] = v_ref
    else:
        desired_config[-state.nv:] = np.zeros(state.nv)

    activation_xreg = crocoddyl.ActivationModelWeightedQuad(w_x**2)
    x_reg_cost = crocoddyl.CostModelResidual(
        state, activation_xreg, crocoddyl.ResidualModelState(state, desired_config, actuation.nu)
    )
    w_u = np.copy(planner_weights.WBC_WEIGHTED_COSTS['uReg'])
    # w_u /= weight_by_mass
    w_u /= weight_by_ulim
    activation_ureg = crocoddyl.ActivationModelWeightedQuad(w_u ** 2)
    u_reg_cost = crocoddyl.CostModelResidual(
        state, activation_ureg, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    costs.addCost("xReg", x_reg_cost, planner_weights.WBC_COST_WEIGHTS['xReg'])
    costs.addCost("uReg", u_reg_cost, planner_weights.WBC_COST_WEIGHTS['uReg'])

    # Adding the state limits penalization
    jvel_lim_safety_margin = 1
    x_lb = np.concatenate([state.lb[1: state.nv + 1], jvel_lim_safety_margin * state.lb[-state.nv:]])
    x_ub = np.concatenate([state.ub[1: state.nv + 1], jvel_lim_safety_margin * state.ub[-state.nv:]])
    activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(x_lb, x_ub)
    )
    x_bounds = crocoddyl.CostModelResidual(
        state,
        activation_xbounds,
        crocoddyl.ResidualModelState(state, nu=actuation.nu),
    )
    costs.addCost("xBounds", x_bounds, planner_weights.WBC_COST_WEIGHTS['xBounds'])

    #
    # Collision Avoidance at Joint Level
    #
    runningConstraintModelManager = crocoddyl.ConstraintModelManager(
        state, actuation.nu
    )
    if b_sqp or (b_sca and geom_model is not None):
        # enforce torque limits as hard constraints
        control_bounds = ControlBounds(state, actuation.nu)
        u_bounds_res = crocoddyl.ConstraintModelResidual(state,
                                          control_bounds,
                                          -ulim,
                                          ulim,
                                          True)
        runningConstraintModelManager.addConstraint('uBounds',
                                                    u_bounds_res,
                                                    True)

        # enforce joint limits as hard constraints
        costs.removeCost("xBounds")
        state_bounds = crocoddyl.ResidualModelState(state, actuation.nu)
        state_bounds_res = crocoddyl.ConstraintModelResidual(state,
                                                            state_bounds,
                                                            x_lb,
                                                            x_ub,
                                                            True)
        runningConstraintModelManager.addConstraint('stateBounds',
                                                    state_bounds_res,
                                                    True)

        if b_sca:
            # Replace the soft friction-cone penalization with a hard constraint
            # on the linear contact force: 4 linearized tangential facets plus a
            # unilateral normal-force row.  Contact moments / CoP of the 6D
            # contacts (feet, knees) remain unconstrained for now.
            if getattr(planner_weights, 'B_HARD_FRICTION_CONE_SCA', True):
                for fr_name, fr_plane in frames_in_contact.items():
                    costs.removeCost(fr_name + "_friction")

                    fr_id = plan_to_model_ids[fr_name]
                    r, p = util.util.vec_to_roll_pitch(fr_plane)
                    plane_rot = util.util.euler_to_rot([r, p, 0])
                    friction_cone = crocoddyl.FrictionCone(plane_rot, mu, 4, True)
                    friction_residual = crocoddyl.ResidualModelContactFrictionCone(
                        state, fr_id, friction_cone, actuation.nu)
                    constr_friction = crocoddyl.ConstraintModelResidual(state,
                                                                        friction_residual,
                                                                        friction_cone.lb,
                                                                        friction_cone.ub,
                                                                        True)
                    runningConstraintModelManager.addConstraint(fr_name + "_friction",
                                                                constr_friction,
                                                                True)

            if True:
                b_soft_col_avoid = False
                for cp_idx, cp in enumerate(geom_model.collisionPairs):
                    cp_first_name = geom_model.geometryObjects[cp.first].name
                    cp_second_name = geom_model.geometryObjects[cp.second].name
                    if 'link_0' in cp_first_name:
                        cp_first_name = cp_first_name.replace('_link_0', '_joint')
                    elif 'primitive_shape' in cp_first_name:
                        cp_first_name = 'root_joint'
                    elif 'hand' in cp_first_name:
                        cp_first_name = cp_first_name.replace('rubber_hand_0', 'wrist_yaw_joint')
                    else:
                        raise ValueError(f'[SCA] Name parsing of {cp_first_name} not specified.')
                    # get joint id of nearest joint to the collision pair
                    j_id = robot_model.getJointId(cp_first_name)

                    # add as cost just for torso (debugging purposes -- remove IF condition later)
                    if (
                            'root_joint' in cp_first_name or 'torso' in cp_second_name      # torso vs all
                            or ('door' in cp_second_name or 'door' in cp_first_name)        # door vs all
                            or ('stairs' in cp_second_name and 'knee' in cp_first_name)     # stairs vs knees
                            or ('stairs' in cp_second_name and 'ankle' in cp_first_name)    # stairs vs ankles
                            # or ('hole' in cp_second_name and 'hip' in cp_first_name)        # hole vs hips
                            # or ('hole' in cp_second_name or 'hole' in cp_first_name)       # hole vs knees
                            or ('hole' in cp_second_name and 'pelvis' in cp_first_name)       # hole vs knees
                            or ('hole' in cp_second_name and 'knee' in cp_first_name)       # hole vs knees
                            or ('hole' in cp_second_name and 'ankle' in cp_first_name)      # hole vs ankles
                            or ('hole' in cp_second_name and 'elbow' in cp_first_name)      # hole vs elbows
                            # or ('hole' in cp_second_name and 'shoulder' in cp_first_name)      # hole vs shoulders
                            or ('right_ankle' in cp_first_name and 'right_hip' in cp_second_name)     # ankle-hip
                            or ('right_hip' in cp_first_name and 'right_ankle' in cp_second_name)     # ankle-hip
                            or ('left_ankle' in cp_first_name and 'left_hip' in cp_second_name)     # ankle-hip
                            or ('left_hip' in cp_first_name and 'left_ankle' in cp_second_name)     # ankle-hip
                            or ('right_ankle' in cp_first_name and 'left_knee' in cp_second_name)     # leg cross
                            or ('left_knee' in cp_first_name and 'right_ankle' in cp_second_name)
                            or ('left_ankle' in cp_first_name and 'right_knee' in cp_second_name)
                            or ('right_knee' in cp_first_name and 'left_ankle' in cp_second_name)
                            or ('right_knee' in cp_first_name and 'left_hip' in cp_second_name)
                            or ('left_hip' in cp_first_name and 'right_knee' in cp_second_name)
                            or ('left_ankle' in cp_first_name and 'right_ankle' in cp_second_name)
                            or ('right_ankle' in cp_first_name and 'left_ankle' in cp_second_name)
                            or ('left_knee' in cp_first_name and 'right_hip' in cp_second_name)  # opposing knee-hip
                            or ('right_hip' in cp_first_name and 'left_knee' in cp_second_name)  # opposing knee-hip
                            or ('right_knee' in cp_first_name and 'left_hip' in cp_second_name)  # opposing knee-hip
                            or ('left_hip' in cp_first_name and 'right_knee' in cp_second_name)  # opposing knee-hip
                            or ('pelvis' in cp_first_name and 'hip' in cp_second_name)  # pelvis-hips
                            or ('hip' in cp_first_name and 'pelvis' in cp_second_name)  # pelvis-hips
                    ):
                        sca_alpha = 0.005
                        dist_col = ResidualDistanceCollision(state, actuation.nu, geom_model, cp_idx)
                        # dist_col = crocoddyl.ResidualModelPairCollision(state, actuation.nu, geom_model, cp_idx, j_id)
                        if b_soft_col_avoid:
                            activation_sca = ActivationModelDistanceQuad(1, 0.4, 0.2)
                            # activation_sca = crocoddyl.ActivationModelQuadFlatExp(1, sca_alpha)
                            # activation_sca = crocoddyl.ActivationModelQuadFlatExp(3, sca_alpha)
                            # activation_sca = crocoddyl.ActivationModelQuadraticBarrier(
                            #     crocoddyl.ActivationBounds(np.array([0.08, 0.08, 0.08]),
                            #                                np.array([1.5, 1.5, 1.5]))
                            # )
                            sca_cost = crocoddyl.CostModelResidual(
                                state,
                                activation_sca,
                                dist_col,
                            )
                            costs.addCost(cp_first_name + '_to_' + cp_second_name + "_sca",
                                          sca_cost,
                                      planner_weights.WBC_COST_WEIGHTS['sca'])
                        else:   # use as hard inequality constraint
                            # False to deactivate at terminal step
                            col_avoid_constr = crocoddyl.ConstraintModelResidual(state,
                                                                                 dist_col,
                                                                                 np.array([0.0]),
                                                                                 np.array([np.inf]),
                                                                                 True)
                            # col_avoid_constr = crocoddyl.ConstraintModelResidual(state,
                            #                                                      dist_col,
                            #                                                      np.array([0.02, 0.02, 0.02]),
                            #                                                      np.array([np.inf, np.inf, np.inf]),
                            #                                                      True)
                            runningConstraintModelManager.addConstraint(cp_first_name + '_to_' + cp_second_name + "_sca",
                                                                        col_avoid_constr,
                                                                        True)


    # Creating the action model
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contacts, costs, runningConstraintModelManager
    )
    return dmodel


def createMultiFrameFinalActionModel(state: crocoddyl.StateMultibody,
                                actuation: crocoddyl.ActuationModelFloatingBase,
                                x0: np.array,
                                plan_to_model_ids: dict[str, int],
                                frames_in_contact: dict[str: np.array],
                                next_frames_in_contact: dict[str: np.array],
                                frame_targets_dict: dict[str, np.array],
                                joint_names_dict: dict[str, list[str]] = None,
                                planner_weights: PlannerConfig | BaselinePlannerConfig = None,
                                zero_config: np.array = None,
                                v_ref: np.array = None,
                                terminal_step: bool = False,
                                robot_model: pinocchio.Model = None) -> crocoddyl.DifferentialActionModelContactFwdDynamics:
    desired_config = np.copy(x0)

    # Define the cost sum (cost manager)
    costs = crocoddyl.CostModelSum(state, actuation.nu)

    # Define contacts (e.g., feet / hand supports)
    contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)

    # scale cost of joint velocities to be higher on limbs in contact
    wx_scale =  np.ones(actuation.nu)

    # create contact models for each frame in contact
    for fr_name, fr_plane in frames_in_contact.items():
        fr_id = plan_to_model_ids[fr_name]

        # for hand contact frames, set the corresponding rotation
        SE3_ee = pin.SE3.Identity()
        # SE3_ee.rotation = so3_from_vec_to_vec(Z_UP, fr_plane)
        r,p = util.util.vec_to_roll_pitch(fr_plane)
        SE3_ee.rotation = util.util.euler_to_rot([r,p,0])

        if 'H' in fr_name:
            fr_contact = crocoddyl.ContactModel3D(
                state,
                fr_id,
                np.zeros(3),
                pin.LOCAL_WORLD_ALIGNED,
                actuation.nu,
                np.array(planner_weights.WBP_BAUMGARTE_GAINS3D),
            )
        else:
            fr_contact = crocoddyl.ContactModel6D(
                state,
                fr_id,
                SE3_ee,
                pin.LOCAL_WORLD_ALIGNED,
                actuation.nu,
                np.array(planner_weights.WBP_BAUMGARTE_GAINS6D),
            )
        contacts.addContact(fr_name + "_contact", fr_contact)

        # Add friction cone penalization according to foot or hand contact
        floor_rotation = np.eye(3)
        if 'H' in fr_name:
            # surf_cone = crocoddyl.FrictionCone(floor_rotation, mu, 4, True)     # better if False?
            surf_cone = crocoddyl.FrictionCone(SE3_ee.rotation, mu, 4, True)     # better if False?
        else:
            foot_size = planner_weights.FOOT_SIZE
            surf_cone = crocoddyl.WrenchCone(SE3_ee.rotation, mu, np.array(foot_size), 4, True)
            # surf_cone = crocoddyl.WrenchCone(floor_rotation, mu, np.array(foot_size), 4, True)

        # friction cone activation function
        surf_activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(surf_cone.lb, surf_cone.ub)
        )
        if 'H' in fr_name:
            fr_friction = crocoddyl.CostModelResidual(
                state,
                surf_activation_friction,
                crocoddyl.ResidualModelContactFrictionCone(state, fr_id, surf_cone, actuation.nu),
            )
        else:
            fr_friction = crocoddyl.CostModelResidual(
                state,
                surf_activation_friction,
                crocoddyl.ResidualModelContactWrenchCone(state, fr_id, surf_cone, actuation.nu),
            )
        # costs.addCost(fr_name + "_friction",
        #               fr_friction,
        #               planner_weights.WBC_FINAL_COST_WEIGHTS['friction'])

        # increase cost of joint velocities on limbs in contact
        if joint_names_dict is not None:
            if 'LF' in fr_name:
                jnt_in_contact = get_limb_joint_idx(joint_names_dict['left_leg'], robot_model)
            elif 'RF' in fr_name:
                jnt_in_contact = get_limb_joint_idx(joint_names_dict['right_leg'], robot_model)
            elif 'LH' in fr_name:
                jnt_in_contact = get_limb_joint_idx(joint_names_dict['left_arm'], robot_model)
            elif 'RH' in fr_name:
                jnt_in_contact = get_limb_joint_idx(joint_names_dict['right_arm'], robot_model)
            wx_scale[jnt_in_contact] *= planner_weights.WBP_CONTACT_JVEL_SCALE

    # Add frame-placement cost
    for fr_name, fr_id in plan_to_model_ids.items():
        # skip if no target is provided for this frame (used mostly for baseline planner)
        if fr_name not in frame_targets_dict.keys():
            continue

        # set higher tracking cost on feet
        if 'F' in fr_name:
            w_fr = planner_weights.WBC_FINAL_FRAME_TRACKING_GAINS['feet']
        elif 'LH' in fr_name:
            w_fr = planner_weights.WBC_FINAL_FRAME_TRACKING_GAINS['LH']
        elif 'RH' in fr_name:
            w_fr = planner_weights.WBC_FINAL_FRAME_TRACKING_GAINS['RH']
        elif 'R_knee' in fr_name:
            w_fr = planner_weights.WBC_FINAL_FRAME_TRACKING_GAINS['R_knee']
        elif 'L_knee' in fr_name:
            w_fr = planner_weights.WBC_FINAL_FRAME_TRACKING_GAINS['L_knee']
        elif 'torso' in fr_name:
            w_fr = planner_weights.WBC_FINAL_FRAME_TRACKING_GAINS['torso']
        else:
            raise ValueError(f"Weights to track frame {fr_name} were not set")

        # set the desired frame pose for feet using upcoming contact surface normal
        fr_Mref = pin.SE3.Identity()
        if fr_name in next_frames_in_contact.keys() and 'H' not in fr_name:
            r, p = util.util.vec_to_roll_pitch(next_frames_in_contact[fr_name])
            fr_Mref.rotation = util.util.euler_to_rot([r, p, 0])
            print(f"[Final Tracking] {fr_name} target: {frame_targets_dict[fr_name]}, (r, p):{r, p}")
        elif fr_name in frames_in_contact.keys() and 'H' not in fr_name:
            r, p = util.util.vec_to_roll_pitch(frames_in_contact[fr_name])
            fr_Mref.rotation = util.util.euler_to_rot([r, p, 0])
            print(f"[Final Tracking] {fr_name} target: {frame_targets_dict[fr_name]}, (r, p):{r, p}")
        fr_Mref.translation = frame_targets_dict[fr_name]

        activation_fr = crocoddyl.ActivationModelWeightedQuad(w_fr ** 2)
        fr_cost = crocoddyl.CostModelResidual(
            state,
            activation_fr,
            crocoddyl.ResidualModelFramePlacement(state, fr_id, fr_Mref, actuation.nu),
        )
        costs.addCost(fr_name + "_goal",
                      fr_cost,
                      planner_weights.WBC_FINAL_COST_WEIGHTS['frame_goal'])

    weight_by_ulim = state.pinocchio.effortLimit[-(state.nv-6):]
    weight_by_mass = [i.mass for i in state.pinocchio.inertias.tolist()[-(state.nv-6):]]

    # Adding state and control regularization terms
    w_x = np.copy(planner_weights.WBC_PHASE_END_WEIGHTED_COSTS['xReg'])
    w_x[-actuation.nu:] *= wx_scale     # penalize more jvel of contact limb
    w_x[6:state.nq-1] *= wx_scale    # penalize more jpos of contact limb

    # w_x[-actuation.nu:] /= weight_by_ulim
    if zero_config is not None and terminal_step:
        w_x = np.copy(planner_weights.WBC_FINAL_WEIGHTED_COSTS['xReg'])
        w_x[-actuation.nu:] *= wx_scale  # penalize more jvel of contact limb
        w_x[6:state.nq - 1] *= wx_scale  # penalize more jpos of contact limb

        # ---- all joints except base (linear) position
        # desired_config[3:7] = np.array([0, 0, 0, 1])
        desired_config[7+14:state.nq] = zero_config[7+14:]
        w_x[:3] = 0         # do not track linear base position
        w_x[7:7+14] = 0     # do not track lower body joints
        # ---- only upper body joints (of G1)
        # desired_config[7+14:state.nq] = zero_config[7+14:]
        # w_x[:7+14] = 0
    if v_ref is not None:
        desired_config[-state.nv:] = v_ref
    else:
        desired_config[-state.nv:] = np.zeros(state.nv)

    # if terminal_step:
    activation_xreg = crocoddyl.ActivationModelWeightedQuad(w_x ** 2)
    x_reg_cost = crocoddyl.CostModelResidual(
        state, activation_xreg, crocoddyl.ResidualModelState(state, desired_config, actuation.nu)
    )
    costs.addCost("xReg",
                  x_reg_cost,
                  planner_weights.WBC_FINAL_COST_WEIGHTS['xReg'])

    # Allow larger control input where torque limits are larger
    w_u = np.copy(planner_weights.WBC_WEIGHTED_COSTS['uReg'])
    # w_u /= weight_by_mass
    w_u /= weight_by_ulim
    activation_ureg = crocoddyl.ActivationModelWeightedQuad(w_u ** 2)
    u_reg_cost = crocoddyl.CostModelResidual(
        state, activation_ureg, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    # crocoddyl.ResidualModelJointEffort
    # costs.addCost("uReg",
    #               u_reg_cost,
    #               planner_weights.WBC_FINAL_COST_WEIGHTS['uReg'])

    # Adding the state limits penalization
    x_lb = np.concatenate([state.lb[1: state.nv + 1], state.lb[-state.nv:]])
    x_ub = np.concatenate([state.ub[1: state.nv + 1], state.ub[-state.nv:]])
    activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(x_lb, x_ub)
    )
    x_bounds = crocoddyl.CostModelResidual(
        state,
        activation_xbounds,
        crocoddyl.ResidualModelState(state, nu=actuation.nu),
    )
    costs.addCost("xBounds", x_bounds, planner_weights.WBC_FINAL_COST_WEIGHTS['xBounds'])

    # Creating the action model
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contacts, costs, 0.0, True
    )
    return dmodel


def createMultiFrameFinalImpulseModel(state: crocoddyl.StateMultibody,
                                      x0: np.array,
                                      plan_to_model_ids: dict[str, int],
                                      frames_in_contact: dict[str: np.array],
                                      next_frames_in_contact: dict[str: np.array],
                                      frame_targets_dict: dict[str, np.array],
                                      planner_weights: PlannerConfig | BaselinePlannerConfig = None,
                                      zero_config: np.array = None,
                                      v_ref: np.array = None) -> crocoddyl.ActionModelImpulseFwdDynamics:
    desired_config = np.copy(x0)

    # Creating a 6D multi-contact model, and then including the supporting foot
    impulseModel = crocoddyl.ImpulseModelMultiple(state)

    for fr_name, fr_plane in next_frames_in_contact.items():
        # apply impulse only to the new (upcoming) contacts
        if fr_name not in frames_in_contact.keys():
            fr_id = plan_to_model_ids[fr_name]

            if 'H' in fr_name:
                supportContactModel = crocoddyl.ImpulseModel3D(
                    state, fr_id, pin.LOCAL_WORLD_ALIGNED
                )
            else:
                supportContactModel = crocoddyl.ImpulseModel6D(
                    state, fr_id, pin.LOCAL_WORLD_ALIGNED
                )
            impulseModel.addImpulse(
                fr_name + "_impulse", supportContactModel
            )

    # Creating the cost model for a contact phase
    costs = crocoddyl.CostModelSum(state, 0)

    # Add frame-placement cost
    for fr_name, fr_id in plan_to_model_ids.items():
        # skip if no target is provided for this frame (used mostly for baseline planner)
        if fr_name not in frame_targets_dict.keys():
            continue

        # set higher tracking cost on feet
        if 'F' in fr_name:
            w_fr = planner_weights.WBC_FRAME_TRACKING_GAINS['feet']
        elif 'LH' in fr_name:
            w_fr = planner_weights.WBC_FRAME_TRACKING_GAINS['LH']
        elif 'RH' in fr_name:
            w_fr = planner_weights.WBC_FRAME_TRACKING_GAINS['RH']
        elif 'R_knee' in fr_name:
            w_fr = planner_weights.WBC_FRAME_TRACKING_GAINS['R_knee']
        elif 'L_knee' in fr_name:
            w_fr = planner_weights.WBC_FRAME_TRACKING_GAINS['L_knee']
        elif 'torso' in fr_name:
            w_fr = planner_weights.WBC_FRAME_TRACKING_GAINS['torso']
        else:
            raise ValueError(f"Weights to track frame {fr_name} were not set")

        # set the desired frame pose
        fr_Mref = pin.SE3.Identity()
        if fr_name in next_frames_in_contact.keys() and 'H' not in fr_name:
            # fr_Mref.rotation = so3_from_vec_to_vec(Z_UP, next_frames_in_contact[fr_name])
            r, p = util.util.vec_to_roll_pitch(next_frames_in_contact[fr_name])
            fr_Mref.rotation = util.util.euler_to_rot([r, p, 0])
        fr_Mref.translation = frame_targets_dict[fr_name]

        activation_fr = crocoddyl.ActivationModelWeightedQuad(w_fr ** 2)
        fr_cost = crocoddyl.CostModelResidual(
            state,
            activation_fr,
            crocoddyl.ResidualModelFramePlacement(state, fr_id, fr_Mref, 0),
        )
        costs.addCost(fr_name + "_goal",
                      fr_cost,
                      planner_weights.WBC_IMPULSE_COST_WEIGHTS['frame_goal'])

    # Adding state and control regularization terms
    w_x = planner_weights.WBC_WEIGHTED_COSTS['xReg']
    if zero_config is not None:     # and terminal_step:
        desired_config[3:state.nq] = zero_config[3:]
        w_x[:3] = 0
    if v_ref is not None:
        desired_config[-state.nv:] = v_ref
    else:
        desired_config[-state.nv:] = np.zeros(state.nv)
    activation_xreg = crocoddyl.ActivationModelWeightedQuad(w_x**2)
    x_reg_cost = crocoddyl.CostModelResidual(
        state, activation_xreg, crocoddyl.ResidualModelState(state, desired_config, 0)
    )
    costs.addCost("xReg",
                  x_reg_cost,
                  planner_weights.WBC_IMPULSE_COST_WEIGHTS['xReg'])

    # stateWeights = np.array(
    #     [1.0] * 6 + [0.1] * (state.nv - 6) + [10] * state.nv
    # )
    # stateResidual = crocoddyl.ResidualModelState(
    #     state, desired_config, 0
    # )
    # stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights ** 2)
    # stateReg = crocoddyl.CostModelResidual(
    #     state, stateActivation, stateResidual
    # )
    # costModel.addCost("stateReg", stateReg, 1e1)

    # Creating the action model for the KKT dynamics with simpletic Euler
    # integration scheme
    dmodel = crocoddyl.ActionModelImpulseFwdDynamics(
        state, impulseModel, costs
    )
    dmodel.JMinvJt_damping = 1e-12
    dmodel.r_coeff = 0.0
    return dmodel


def createSequence(dmodels, DT, N, integration_type='Euler'):
    control = crocoddyl.ControlParametrizationModelPolyOne(dmodels[0].actuation.nu)
    if integration_type == 'Euler':
        return [
            [crocoddyl.IntegratedActionModelEuler(m, DT)] * N
            for m in dmodels
        ]
    elif integration_type == 'RK2':
        return [
            [crocoddyl.IntegratedActionModelRK(m, control, crocoddyl.RKType.two, DT)] * N
            for m in dmodels
        ]
    else:
        raise ValueError(f"Integration type {integration_type} not recognized.")

def createFinalSequence(dmodels, integration_type='Euler'):
    control = crocoddyl.ControlParametrizationModelPolyOne(dmodels[0].actuation.nu)
    if integration_type == 'Euler':
        return [
            [crocoddyl.IntegratedActionModelEuler(m, 0)]
            for m in dmodels
        ]
    elif integration_type == 'RK2':
        return [
            [crocoddyl.IntegratedActionModelRK(m, control, crocoddyl.RKType.two, 0)]
            for m in dmodels
        ]
    else:
        raise ValueError(f"Integration type {integration_type} not recognized.")



# --------
# Utilities
# --------
def quasi_static_ocp(frames_in_contact: dict[str: np.ndarray],
                     plan_to_model_ids: dict[str: int],
                     pin_model: pinocchio.Model,
                     x0: np.ndarray, ):
    import cvxpy as cp
    # max allowed normal force of some factor x robot weight
    mu = 0.7

    # construct equality constraints from Centroidal Dynamics
    lf_frame_id = plan_to_model_ids['LF']
    lf_joint_id = pin_model.frames[lf_frame_id].parentJoint
    rf_frame_id = plan_to_model_ids['RF']
    rf_joint_id = pin_model.frames[rf_frame_id].parentJoint
    pin_data = pin_model.createData()
    pin.forwardKinematics(pin_model, pin_data, x0[:pin_model.nq])
    pin.updateFramePlacements(pin_model, pin_data)

    # get positions of the hands
    lh_frame_id = plan_to_model_ids['LH']
    lh_joint_id = pin_model.frames[lh_frame_id].parentJoint
    rh_frame_id = plan_to_model_ids['RH']
    rh_joint_id = pin_model.frames[rh_frame_id].parentJoint

    # get positions of the feet and hands
    lf_placement = pin.updateFramePlacement(pin_model, pin_data, lf_frame_id)
    lf_pos = lf_placement.translation
    rf_placement = pin.updateFramePlacement(pin_model, pin_data, rf_frame_id)
    rf_pos = rf_placement.translation
    lh_placement = pin.updateFramePlacement(pin_model, pin_data, lh_frame_id)
    lh_pos = lh_placement.translation
    rh_placement = pin.updateFramePlacement(pin_model, pin_data, rh_frame_id)
    rh_pos = rh_placement.translation

    # get center of mass position
    com_pos = pin.centerOfMass(pin_model, pin_data, x0[:pin_model.nq])

    delta_lf = util.liegroup.VecToso3(lf_pos - com_pos)
    delta_rf = util.liegroup.VecToso3(rf_pos - com_pos)
    delta_lh = util.liegroup.VecToso3(lh_pos - com_pos)
    delta_rh = util.liegroup.VecToso3(rh_pos - com_pos)

    # assume point contacts for simplicity
    num_contacts = len(frames_in_contact)
    floor_friction_submat = np.zeros((6, 3))
    lwall_friction_submat = np.zeros((7, 3))
    rwall_friction_submat = np.zeros((7, 3))
    A_ineq = []
    b_ineq = []
    A_eq = np.zeros((6, 3 * num_contacts))
    b_eq = np.zeros((6, 1))
    b_eq[2] = pin_data.mass[0] * 9.81

    # create friction sub-matrix with linearized constraints
    #------------ floor: normal in +z
    floor_friction_submat[0, :] = np.array([1, 0, -mu])
    floor_friction_submat[1, :] = np.array([-1, 0, -mu])
    floor_friction_submat[2, :] = np.array([0, 1, -mu])
    floor_friction_submat[3, :] = np.array([0, -1, -mu])
    floor_friction_submat[4, :] = np.array([0, 0, -1])  # positive z force
    floor_friction_submat[5, :] = np.array([0, 0, 1])   # max normal force bounded by weight
    #------------ left wall: normal in -y
    lwall_friction_submat[0, :] = np.array([1, mu, 0])
    lwall_friction_submat[1, :] = np.array([-1, mu, 0])
    lwall_friction_submat[2, :] = np.array([0, mu, 1])
    lwall_friction_submat[3, :] = np.array([0, mu, -1])
    lwall_friction_submat[4, :] = np.array([0, 1, 0])
    lwall_friction_submat[5, :] = np.array([0, -1, 0])  # max normal force bounded by weight
    lwall_friction_submat[6, :] = np.array([0, 0, -1])  # choose positive z force
    #------------ right wall: normal in +y
    rwall_friction_submat[0, :] = np.array([1, -mu, 0])
    rwall_friction_submat[1, :] = np.array([-1, -mu, 0])
    rwall_friction_submat[2, :] = np.array([0, -mu, 1])
    rwall_friction_submat[3, :] = np.array([0, -mu, -1])
    rwall_friction_submat[4, :] = np.array([0, -1, 0])
    rwall_friction_submat[5, :] = np.array([0, 1, 0])   # max normal force bounded by weight
    rwall_friction_submat[6, :] = np.array([0, 0, -1])  # choose positive z force

    # fill equality constraint matrices considering all contacts
    contact_joint_ids = []
    for i, (fr_name, fr_plane) in enumerate(frames_in_contact.items()):
        A_eq[0:3, i * 3:(i + 1) * 3] = np.eye(3)

        if fr_name == 'LF':
            current_friction_submat = np.zeros((6, 3 * num_contacts))
            r_hat = delta_lf
            contact_joint_ids.append(lf_joint_id)
            current_friction_submat[:, 3*i:3*(i+1)] = floor_friction_submat
            fz_max = pin.computeTotalMass(pin_model) * 9.81
            current_friction_subvec = np.array([[0], [0], [0], [0], [0], [fz_max]])
        elif fr_name == 'RF':
            current_friction_submat = np.zeros((6, 3 * num_contacts))
            r_hat = delta_rf
            contact_joint_ids.append(rf_joint_id)
            current_friction_submat[:, 3*i:3*(i+1)] = floor_friction_submat
            fz_max = pin.computeTotalMass(pin_model) * 9.81
            current_friction_subvec = np.array([[0], [0], [0], [0], [0], [fz_max]])
        elif fr_name == 'LH':
            current_friction_submat = np.zeros((7, 3 * num_contacts))
            r_hat = delta_lh
            contact_joint_ids.append(lh_joint_id)
            current_friction_submat[:, 3*i:3*(i+1)] = lwall_friction_submat
            fz_max = pin.computeTotalMass(pin_model) * 9.81 * 0.15 # hands take less load
            current_friction_subvec = np.array([[0], [0], [0], [0], [0], [fz_max], [0]])
        elif fr_name == 'RH':
            current_friction_submat = np.zeros((7, 3 * num_contacts))
            r_hat = delta_rh
            contact_joint_ids.append(rh_joint_id)
            current_friction_submat[:, 3*i:3*(i+1)] = rwall_friction_submat
            fz_max = pin.computeTotalMass(pin_model) * 9.81 * 0.15 # hands take less load
            current_friction_subvec = np.array([[0], [0], [0], [0], [0], [fz_max], [0]])
        else:
            raise ValueError(f"Contact {fr_name} not recognized for quasi-static LP equality constraint.")
        A_eq[3:6, i * 3:(i + 1) * 3] = r_hat

        # populate inequality constraint matrices considering all contacts
        A_ineq.append(current_friction_submat)
        b_ineq.append(current_friction_subvec)

    # create and solve optimization problem
    f = cp.Variable((3 * num_contacts, 1))   # decision variable: contact forces
    # minimize static sum of moments without reaction torques
    objective = cp.Minimize(cp.norm(A_eq[3:6,:] @ f - b_eq[3:6]))   #  + cp.norm(f)
    A_ineq = np.concatenate(A_ineq)
    b_ineq = np.concatenate(b_ineq)
    constraints = [A_eq[:3,:] @ f == b_eq[:3], A_ineq @ f <= b_ineq]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver='CLARABEL', verbose=False)
    if prob.status != 'optimal':
        raise ValueError('[quasi_static_lp] Contact force optimization problem not solved optimally.')

    # Convert pin_forces from world frame to local frame at the respective joints
    pin_forces = pin.StdVec_Force(pin_model.njoints, pin.Force.Zero())
    for con_idx, j_idx in enumerate(contact_joint_ids):
        curr_f = f.value[3 * con_idx:3 * (con_idx + 1)]     # current fx, fy, fz
        wrench = pin.Force(np.vstack((curr_f, np.zeros((3, 1)))))    # we assumed only forces
        joint_placement = pin_data.oMi[j_idx]
        pin_forces[j_idx] = joint_placement.actInv(wrench)
        pin_forces[j_idx].angular = np.zeros(3)   # ignore torques for quasi-static

    static_torques = pin.rnea(pin_model, pin_data, x0[:pin_model.nq], np.zeros(pin_model.nv), np.zeros(pin_model.nv),
                              pin_forces)[6:]

    return static_torques