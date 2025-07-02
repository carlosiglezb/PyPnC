import numpy as np
import crocoddyl
import pinocchio as pin

from config.multicontact.planner_config import PlannerConfig
from util.util import so3_from_vec_to_vec

Z_UP =  np.array([0., 0., 1])


def createMultiFrameActionModel(state: crocoddyl.StateMultibody,
                                actuation: crocoddyl.ActuationModelFloatingBase,
                                x0: np.array,
                                plan_to_model_ids: dict[str, int],
                                frames_in_contact: dict[str: np.array],
                                next_frames_in_contact: dict[str: np.array],
                                frame_targets_dict: dict[str, np.array],
                                planner_weights: PlannerConfig = None,
                                zero_config: np.array = None,
                                v_ref: np.array = None,
                                terminal_step: bool = False):
    mu = 0.7

    # Define the cost sum (cost manager)
    costs = crocoddyl.CostModelSum(state, actuation.nu)

    # Define contacts (e.g., feet / hand supports)
    contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)

    # create contact models for each frame in contact
    for fr_name, fr_plane in frames_in_contact.items():
        fr_id = plan_to_model_ids[fr_name]

        # set the corresponding rotation for all frames in contact
        SE3_ee = pin.SE3.Identity()
        SE3_ee.rotation = so3_from_vec_to_vec(Z_UP, fr_plane)

        fr_contact = crocoddyl.ContactModel6D(
            state,
            fr_id,
            SE3_ee,
            pin.LOCAL_WORLD_ALIGNED,
            actuation.nu,
            np.array([0, 1e-6]),
        )
        contacts.addContact(fr_name + "_contact", fr_contact)

        # Add friction cone penalization according to foot or hand contact
        floor_rotation = np.eye(3)
        surf_cone = crocoddyl.FrictionCone(floor_rotation, mu, 4, True)     # better if False?

        # friction cone activation function
        surf_activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(surf_cone.lb, surf_cone.ub)
        )
        fr_friction = crocoddyl.CostModelResidual(
            state,
            surf_activation_friction,
            crocoddyl.ResidualModelContactFrictionCone(state, fr_id, surf_cone, actuation.nu),
        )
        costs.addCost(fr_name + "_friction",
                      fr_friction,
                      planner_weights.WBC_COST_WEIGHTS['friction'])

    # Add frame-placement cost
    for fr_name, fr_id in plan_to_model_ids.items():
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

        # set the desired frame position (and orientation fo upcoming feet contact planes)
        fr_Mref = pin.SE3.Identity()
        if fr_name in next_frames_in_contact.keys() and 'H' not in fr_name:
            fr_Mref.rotation = so3_from_vec_to_vec(Z_UP, next_frames_in_contact[fr_name])
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

    # Adding state and control regularization terms
    if zero_config is not None and terminal_step:
        x0[3:state.nq] = zero_config[3:]
    if v_ref is not None:
        x0[-state.nv:] = v_ref
    else:
        x0[-state.nv:] = np.zeros(state.nv)
    w_x = planner_weights.WBC_WEIGHTED_COSTS['xReg']
    activation_xreg = crocoddyl.ActivationModelWeightedQuad(w_x**2)
    x_reg_cost = crocoddyl.CostModelResidual(
        state, activation_xreg, crocoddyl.ResidualModelState(state, x0, actuation.nu)
    )
    w_u = planner_weights.WBC_WEIGHTED_COSTS['uReg']
    activation_ureg = crocoddyl.ActivationModelWeightedQuad(w_u ** 2)
    u_reg_cost = crocoddyl.CostModelResidual(
        state, activation_ureg, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    costs.addCost("xReg", x_reg_cost, planner_weights.WBC_COST_WEIGHTS['xReg'])
    costs.addCost("uReg", u_reg_cost, planner_weights.WBC_COST_WEIGHTS['uReg'])

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
    costs.addCost("xBounds", x_bounds, planner_weights.WBC_COST_WEIGHTS['xBounds'])

    # Creating the action model
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contacts, costs
    )
    return dmodel


def createMultiFrameFinalActionModel(state: crocoddyl.StateMultibody,
                                actuation: crocoddyl.ActuationModelFloatingBase,
                                x0: np.array,
                                plan_to_model_ids: dict[str, int],
                                frames_in_contact: dict[str: np.array],
                                next_frames_in_contact: dict[str: np.array],
                                frame_targets_dict: dict[str, np.array],
                                planner_weights: PlannerConfig = None,
                                zero_config: np.array = None,
                                v_ref: np.array = None,
                                terminal_step: bool = False):
    mu = 0.7

    # Define the cost sum (cost manager)
    costs = crocoddyl.CostModelSum(state, actuation.nu)

    # Define contacts (e.g., feet / hand supports)
    contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)

    # create contact models for each frame in contact
    for fr_name, fr_plane in frames_in_contact.items():
        fr_id = plan_to_model_ids[fr_name]

        # for hand contact frames, set the corresponding rotation
        SE3_ee = pin.SE3.Identity()
        SE3_ee.rotation = so3_from_vec_to_vec(Z_UP, fr_plane)

        fr_contact = crocoddyl.ContactModel6D(
            state,
            fr_id,
            SE3_ee,
            pin.LOCAL_WORLD_ALIGNED,
            actuation.nu,
            np.array([0, 1e-6]),
        )
        contacts.addContact(fr_name + "_contact", fr_contact)

        # Add friction cone penalization according to foot or hand contact
        floor_rotation = np.eye(3)
        surf_cone = crocoddyl.FrictionCone(floor_rotation, mu, 4, True)     # better if False?

        # friction cone activation function
        surf_activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(surf_cone.lb, surf_cone.ub)
        )
        fr_friction = crocoddyl.CostModelResidual(
            state,
            surf_activation_friction,
            crocoddyl.ResidualModelContactFrictionCone(state, fr_id, surf_cone, actuation.nu),
        )
        costs.addCost(fr_name + "_friction",
                      fr_friction,
                      planner_weights.WBC_FINAL_COST_WEIGHTS['friction'])

    # Add frame-placement cost
    for fr_name, fr_id in plan_to_model_ids.items():
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
            # if zero_config is not None:
            #     w_fr = np.array([0.1] * 3 + [0.01] * 3)
        else:
            raise ValueError(f"Weights to track frame {fr_name} were not set")

        # set the desired frame pose for feet using upcoming contact surface normal
        fr_Mref = pin.SE3.Identity()
        if fr_name in next_frames_in_contact.keys() and 'H' not in fr_name:
            fr_Mref.rotation = so3_from_vec_to_vec(Z_UP, next_frames_in_contact[fr_name])
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

    # Adding state and control regularization terms
    w_x = planner_weights.WBC_WEIGHTED_COSTS['xReg']
    if zero_config is not None and terminal_step:
        x0[3:state.nq] = zero_config[3:]
    if v_ref is not None:
        x0[-state.nv:] = v_ref
    else:
        x0[-state.nv:] = np.zeros(state.nv)

    if terminal_step:
        activation_xreg = crocoddyl.ActivationModelWeightedQuad(w_x ** 2)
        x_reg_cost = crocoddyl.CostModelResidual(
            state, activation_xreg, crocoddyl.ResidualModelState(state, x0, actuation.nu)
        )
        costs.addCost("xReg",
                      x_reg_cost,
                      planner_weights.WBC_FINAL_COST_WEIGHTS['xReg'])

    w_u = planner_weights.WBC_WEIGHTED_COSTS['uReg']
    activation_ureg = crocoddyl.ActivationModelWeightedQuad(w_u ** 2)
    u_reg_cost = crocoddyl.CostModelResidual(
        state, activation_ureg, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    costs.addCost("uReg",
                  u_reg_cost,
                  planner_weights.WBC_FINAL_COST_WEIGHTS['uReg'])

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
                                      planner_weights: PlannerConfig = None,
                                      zero_config: np.array = None,
                                      v_ref: np.array = None):
    # Creating a 6D multi-contact model, and then including the supporting foot
    impulseModel = crocoddyl.ImpulseModelMultiple(state)

    for fr_name, fr_plane in next_frames_in_contact.items():
        # apply impulse only to the new (upcoming) contacts
        if fr_name not in frames_in_contact.keys():
            fr_id = plan_to_model_ids[fr_name]

            SE3_ee = pin.SE3.Identity()
            SE3_ee.rotation = so3_from_vec_to_vec(Z_UP, fr_plane)

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
            # if zero_config is not None:
            #     w_fr = np.array([0.1] * 3 + [0.01] * 3)
        else:
            raise ValueError(f"Weights to track frame {fr_name} were not set")

        # set the desired frame pose
        fr_Mref = pin.SE3.Identity()
        if fr_name in next_frames_in_contact.keys() and 'H' not in fr_name:
            fr_Mref.rotation = so3_from_vec_to_vec(Z_UP, next_frames_in_contact[fr_name])
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
    if zero_config is not None:     # and terminal_step:
        x0[3:state.nq] = zero_config[3:]
    if v_ref is not None:
        x0[-state.nv:] = v_ref
    else:
        x0[-state.nv:] = np.zeros(state.nv)
    w_x = planner_weights.WBC_WEIGHTED_COSTS['xReg']
    activation_xreg = crocoddyl.ActivationModelWeightedQuad(w_x**2)
    x_reg_cost = crocoddyl.CostModelResidual(
        state, activation_xreg, crocoddyl.ResidualModelState(state, x0, 0)
    )
    costs.addCost("xReg",
                  x_reg_cost,
                  planner_weights.WBC_IMPULSE_COST_WEIGHTS['xReg'])

    # stateWeights = np.array(
    #     [1.0] * 6 + [0.1] * (state.nv - 6) + [10] * state.nv
    # )
    # stateResidual = crocoddyl.ResidualModelState(
    #     state, x0, 0
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
    return dmodel


def createSequence(dmodels, DT, N):
    # control = crocoddyl.ControlParametrizationModelPolyOne(dmodels[0].actuation.nu)
    return [
        [crocoddyl.IntegratedActionModelEuler(m, DT)] * N
        for m in dmodels
    ]
    # return [
    #     [crocoddyl.IntegratedActionModelRK(m, control, crocoddyl.RKType.two, DT)] * N
    #     for m in dmodels
    # ]

def createFinalSequence(dmodels):
    # control = crocoddyl.ControlParametrizationModelPolyOne(dmodels[0].actuation.nu)
    return [
        [crocoddyl.IntegratedActionModelEuler(m, 0)]
        for m in dmodels
    ]
    # return [
    #     [crocoddyl.IntegratedActionModelRK(m, control, crocoddyl.RKType.two, 0)]
    #     for m in dmodels
    # ]
