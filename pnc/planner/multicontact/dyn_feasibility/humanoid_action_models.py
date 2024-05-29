import numpy as np
import crocoddyl
import pinocchio as pin
import util.util
from pnc.planner.multicontact.crocoddyl.ResidualModelStateError import ResidualModelStateError


def createDoubleSupportActionModel(state: crocoddyl.StateMultibody,
                                   actuation: crocoddyl.ActuationModelFloatingBase,
                                   x0: np.array,
                                   lf_id: int,
                                   rf_id: int,
                                   lh_id: int,
                                   rh_id: int,
                                   rcj_constraints: crocoddyl.ConstraintModelManager,
                                   lh_target=None,
                                   rh_target=None,
                                   N_horizon=1):
    # Creating a double-support contact (feet support)
    contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)
    lf_contact = crocoddyl.ContactModel6D(
        state,
        lf_id,
        pin.SE3.Identity(),
        pin.LOCAL_WORLD_ALIGNED,
        actuation.nu,
        np.array([0., 0.4]),
    )
    rf_contact = crocoddyl.ContactModel6D(
        state,
        rf_id,
        pin.SE3.Identity(),
        pin.LOCAL_WORLD_ALIGNED,
        actuation.nu,
        np.array([0., 0.4]),
    )
    contacts.addContact("lf_contact", lf_contact)
    contacts.addContact("rf_contact", rf_contact)
    # contact_data = contacts.createData(rob_data)

    # Define the cost sum (cost manager)
    costs = crocoddyl.CostModelSum(state, actuation.nu)

    # Adding the hand-placement cost
    if lh_target is not None:
        w_lhand = np.array([1.] * 3 + [0.00001] * 3)        # (lin, ang)
        lh_Mref = pin.SE3(np.eye(3), lh_target)
        activation_lhand = crocoddyl.ActivationModelWeightedQuad(w_lhand**2)
        lh_cost = crocoddyl.CostModelResidual(
            state,
            activation_lhand,
            crocoddyl.ResidualModelFramePlacement(state, lh_id, lh_Mref, actuation.nu),
        )
        costs.addCost("lh_goal", lh_cost, 1e2)

    if rh_target is not None:
        w_rhand = np.array([1.] * 3 + [0.00001] * 3)        # (lin, ang)
        rh_Mref = pin.SE3(np.eye(3), rh_target)
        activation_rhand = crocoddyl.ActivationModelWeightedQuad(w_rhand**2)
        rh_cost = crocoddyl.CostModelResidual(
            state,
            activation_rhand,
            crocoddyl.ResidualModelFramePlacement(state, rh_id, rh_Mref, actuation.nu),
        )
        costs.addCost("rh_goal", rh_cost, 1e2)

    # Adding state and control regularization terms
    w_x = np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv)
    activation_xreg = crocoddyl.ActivationModelWeightedQuad(w_x**2)
    x_reg_cost = crocoddyl.CostModelResidual(
        state, activation_xreg, crocoddyl.ResidualModelState(state, x0, actuation.nu)
    )
    u_reg_cost = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    costs.addCost("xReg", x_reg_cost, 1e-3)
    costs.addCost("uReg", u_reg_cost, 1e-6)

    # Adding the state limits penalization
    x_lb = np.concatenate([state.lb[1 : state.nv + 1], state.lb[-state.nv :]])
    x_ub = np.concatenate([state.ub[1 : state.nv + 1], state.ub[-state.nv :]])
    activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(x_lb, x_ub)
    )
    x_bounds = crocoddyl.CostModelResidual(
        state,
        activation_xbounds,
        crocoddyl.ResidualModelState(state, 0 * x0, actuation.nu),
    )
    costs.addCost("xBounds", x_bounds, 1.0)

    # Adding the friction cone penalization
    nsurf, mu = np.array([0, 0, 1]), 0.7
    surf_rotation = np.identity(3)
    cone = crocoddyl.FrictionCone(surf_rotation, mu, 4, False)
    activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(cone.lb, cone.ub)
    )
    lf_friction = crocoddyl.CostModelResidual(
        state,
        activation_friction,
        crocoddyl.ResidualModelContactFrictionCone(state, lf_id, cone, actuation.nu),
    )
    rf_friction = crocoddyl.CostModelResidual(
        state,
        activation_friction,
        crocoddyl.ResidualModelContactFrictionCone(state, rf_id, cone, actuation.nu),
    )
    costs.addCost("lf_friction", lf_friction, 1e1)
    costs.addCost("rf_friction", rf_friction, 1e1)

    if rcj_constraints is not None:
        # Add the rolling contact joint constraint as cost
        w_rcj = np.array([0.01])        # (lin, ang)
        l_activation_rcj = crocoddyl.ActivationModelWeightedQuad(w_rcj**2)
        l_rcj_residual = ResidualModelStateError(state, 1, nu=actuation.nu)
        l_rcj_residual.constr_ids = [43, 44]  # left and right foot# left side
        l_rcj_cost = crocoddyl.CostModelResidual(
            state,
            l_activation_rcj,
            l_rcj_residual,
        )
        costs.addCost("l_rcj_cost", l_rcj_cost, 1e-2)

        w_rcj = np.array([0.01])        # (lin, ang)
        r_activation_rcj = crocoddyl.ActivationModelWeightedQuad(w_rcj**2)
        r_rcj_residual = ResidualModelStateError(state, 1, nu=actuation.nu)
        r_rcj_residual.constr_ids = [57, 58]  # right and right foot# left side
        r_rcj_cost = crocoddyl.CostModelResidual(
            state,
            r_activation_rcj,
            r_rcj_residual,
        )
        costs.addCost("r_rcj_cost", r_rcj_cost, 1e-2)

    # Creating the action model
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contacts, costs#, rcj_constraints
    )
    return dmodel


def createSingleSupportHandActionModel(state: crocoddyl.StateMultibody,
                                       actuation: crocoddyl.ActuationModelFloatingBase,
                                       x0: np.array,
                                       lf_id: int,
                                       rf_id: int,
                                       lh_id: int,
                                       rh_id: int,
                                       base_id: int,
                                       base_target,
                                       lfoot_target=None,
                                       rfoot_target=None,
                                       lhand_target=None,
                                       lh_contact=False,
                                       lh_rpy=None,
                                       rhand_target=None,
                                       rh_contact=False,
                                       rh_rpy=None,
                                       ang_weights=0.00001):
    if lh_rpy is None:
        lh_rpy = [0., 0., 0.]

    if rh_rpy is None:
        rh_rpy = [0., 0., 0.]

    # Define the cost sum (cost manager)
    costs = crocoddyl.CostModelSum(state, actuation.nu)

    # Define contacts (e.g., feet /hand supports)
    contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)

    floor_rotation, mu = np.eye(3), 0.7
    floor_cone = crocoddyl.FrictionCone(floor_rotation, mu, 4, False)
    floor_activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(floor_cone.lb, floor_cone.ub)
    )

    # if foot is not moving (e.g., tracking some trajectory), set it as in contact
    if rfoot_target is None:
        rf_contact = crocoddyl.ContactModel6D(
            state,
            rf_id,
            pin.SE3.Identity(),
            pin.LOCAL_WORLD_ALIGNED,
            actuation.nu,
            np.array([0, 0]),
        )
        contacts.addContact("rf_contact", rf_contact)

        rf_friction = crocoddyl.CostModelResidual(
            state,
            floor_activation_friction,
            crocoddyl.ResidualModelContactFrictionCone(state, rf_id, floor_cone, actuation.nu),
        )
        costs.addCost("rf_friction", rf_friction, 1e1)

    # if foot is not moving (e.g., tracking some trajectory), set it as in contact
    if lfoot_target is None:
        lf_contact = crocoddyl.ContactModel6D(
            state,
            lf_id,
            pin.SE3.Identity(),
            pin.LOCAL_WORLD_ALIGNED,
            actuation.nu,
            np.array([0, 0]),
        )
        contacts.addContact("lf_contact", lf_contact)

        lf_friction = crocoddyl.CostModelResidual(
            state,
            floor_activation_friction,
            crocoddyl.ResidualModelContactFrictionCone(state, lf_id, floor_cone, actuation.nu),
        )
        costs.addCost("lf_friction", lf_friction, 1e1)

    # hand is in contact if directly specified
    if lh_contact is True:
        surf_rotation = util.util.euler_to_rot(lh_rpy)
        SE3_lh = pin.SE3.Identity()
        SE3_lh.rotation = surf_rotation

        lh_contact = crocoddyl.ContactModel6D(
            state,
            lh_id,
            SE3_lh,
            pin.LOCAL_WORLD_ALIGNED,
            actuation.nu,
            np.array([0, 0]),
        )
        contacts.addContact("lh_contact", lh_contact)

        # Adding the friction cone penalization
        wall_cone = crocoddyl.FrictionCone(surf_rotation, mu, 4, True)
        wall_activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(wall_cone.lb, wall_cone.ub)
        )
        lh_friction = crocoddyl.CostModelResidual(
            state,
            wall_activation_friction,
            crocoddyl.ResidualModelContactFrictionCone(state, lh_id, wall_cone, actuation.nu),
        )
        costs.addCost("lh_friction", lh_friction, 1e1)

    # hand is in contact if directly specified
    if rh_contact is True:
        surf_rotation = util.util.euler_to_rot(rh_rpy)
        SE3_rh = pin.SE3.Identity()
        SE3_rh.rotation = surf_rotation

        rh_contact = crocoddyl.ContactModel6D(
            state,
            rh_id,
            SE3_rh,
            pin.LOCAL_WORLD_ALIGNED,
            actuation.nu,
            np.array([0, 0]),
        )
        contacts.addContact("rh_contact", rh_contact)

        # Adding the friction cone penalization
        wall_cone = crocoddyl.FrictionCone(surf_rotation, mu, 4, True)
        wall_activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(wall_cone.lb, wall_cone.ub)
        )
        rh_friction = crocoddyl.CostModelResidual(
            state,
            wall_activation_friction,
            crocoddyl.ResidualModelContactFrictionCone(state, rh_id, wall_cone, actuation.nu),
        )
        costs.addCost("rh_friction", rh_friction, 1e1)

    # Add the base-placement cost
    w_base = np.array([1.] * 3 + [ang_weights] * 3)        # (lin, ang)
    base_Mref = pin.SE3(np.eye(3), base_target)
    activation_base = crocoddyl.ActivationModelWeightedQuad(w_base**2)
    base_cost = crocoddyl.CostModelResidual(
        state,
        activation_base,
        crocoddyl.ResidualModelFramePlacement(state, base_id, base_Mref, actuation.nu),
    )
    costs.addCost("base_goal", base_cost, 1e2)

    # Add the foot-placement cost
    if lfoot_target is not None:
        w_lfoot = np.array([1.] * 3 + [1.] * 3)        # (lin, ang)
        lf_Mref = pin.SE3(np.eye(3), lfoot_target)
        activation_lfoot = crocoddyl.ActivationModelWeightedQuad(w_lfoot**2)
        lf_cost = crocoddyl.CostModelResidual(
            state,
            activation_lfoot,
            crocoddyl.ResidualModelFramePlacement(state, lf_id, lf_Mref, actuation.nu),
        )
        costs.addCost("lf_goal", lf_cost, 1e2)

    if rfoot_target is not None:
        w_rfoot = np.array([1.] * 3 + [1.] * 3)        # (lin, ang)
        rf_Mref = pin.SE3(np.eye(3), rfoot_target)
        activation_rfoot = crocoddyl.ActivationModelWeightedQuad(w_rfoot**2)
        rf_cost = crocoddyl.CostModelResidual(
            state,
            activation_rfoot,
            crocoddyl.ResidualModelFramePlacement(state, rf_id, rf_Mref, actuation.nu),
        )
        costs.addCost("rf_goal", rf_cost, 1e2)

    # Adding the hand-placement cost
    if lhand_target is not None:
        w_lhand = np.array([1.] * 3 + [0.00001] * 3)        # (lin, ang)
        lh_Mref = pin.SE3(np.eye(3), lhand_target)
        activation_lhand = crocoddyl.ActivationModelWeightedQuad(w_lhand**2)
        lh_cost = crocoddyl.CostModelResidual(
            state,
            activation_lhand,
            crocoddyl.ResidualModelFramePlacement(state, lh_id, lh_Mref, actuation.nu),
        )
        costs.addCost("lh_goal", lh_cost, 1e2)

    if rhand_target is not None:
        w_rhand = np.array([1.] * 3 + [0.00001] * 3)        # (lin, ang)
        rh_Mref = pin.SE3(np.eye(3), rhand_target)
        activation_rhand = crocoddyl.ActivationModelWeightedQuad(w_rhand**2)
        rh_cost = crocoddyl.CostModelResidual(
            state,
            activation_rhand,
            crocoddyl.ResidualModelFramePlacement(state, rh_id, rh_Mref, actuation.nu),
        )
        costs.addCost("rh_goal", rh_cost, 1e2)

    # Adding state and control regularization terms
    w_x = np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv)
    activation_xreg = crocoddyl.ActivationModelWeightedQuad(w_x**2)
    x_reg_cost = crocoddyl.CostModelResidual(
        state, activation_xreg, crocoddyl.ResidualModelState(state, x0, actuation.nu)
    )
    u_reg_cost = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    costs.addCost("xReg", x_reg_cost, 1e-3)
    costs.addCost("uReg", u_reg_cost, 1e-8)

    # Adding the state limits penalization
    x_lb = np.concatenate([state.lb[1 : state.nv + 1], state.lb[-state.nv :]])
    x_ub = np.concatenate([state.ub[1 : state.nv + 1], state.ub[-state.nv :]])
    activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(x_lb, x_ub)
    )
    x_bounds = crocoddyl.CostModelResidual(
        state,
        activation_xbounds,
        crocoddyl.ResidualModelState(state, 0 * x0, actuation.nu),
    )
    costs.addCost("xBounds", x_bounds, 1.0)

    # Creating the action model
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contacts, costs
    )
    return dmodel


def createMultiFrameActionModel(state: crocoddyl.StateMultibody,
                                actuation: crocoddyl.ActuationModelFloatingBase,
                                x0: np.array,
                                plan_to_model_ids: dict[str, int],
                                frames_in_contact: list[str],
                                ee_rpy: dict[str, list[float]],
                                frame_targets_dict: dict[str, np.array],
                                rcj_constraints: crocoddyl.ConstraintModelManager,
                                zero_config: np.array = None,
                                ang_weights=5.0):
    mu = 0.9

    # Define the cost sum (cost manager)
    costs = crocoddyl.CostModelSum(state, actuation.nu)

    # Define contacts (e.g., feet / hand supports)
    contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)

    # create contact models for each frame in contact
    for fr_name in frames_in_contact:
        fr_id = plan_to_model_ids[fr_name]

        # for hand contact frames, set the corresponding rotation
        SE3_ee = pin.SE3.Identity()
        if fr_name in ee_rpy.keys():
            SE3_ee.rotation = util.util.euler_to_rot(ee_rpy[fr_name])

        fr_contact = crocoddyl.ContactModel6D(
            state,
            fr_id,
            SE3_ee,
            pin.LOCAL_WORLD_ALIGNED,
            actuation.nu,
            np.array([0, 0]),
        )
        contacts.addContact(fr_name, fr_contact)

        # Add friction cone penalization according to foot or hand contact
        if 'RH' in fr_name:
            surf_cone = crocoddyl.FrictionCone(util.util.euler_to_rot(np.array([0., 0., -np.pi/2])), mu, 4, True)
        elif 'LH' in fr_name:
            surf_cone = crocoddyl.FrictionCone(util.util.euler_to_rot(np.array([0., 0., np.pi/2])), mu, 4, True)
        else:
            floor_rotation = np.eye(3)
            surf_cone = crocoddyl.FrictionCone(floor_rotation, mu, 4, False)     # better if False?

        # friction cone activation function
        surf_activation_friction = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(surf_cone.lb, surf_cone.ub)
        )
        fr_friction = crocoddyl.CostModelResidual(
            state,
            surf_activation_friction,
            crocoddyl.ResidualModelContactFrictionCone(state, fr_id, surf_cone, actuation.nu),
        )
        costs.addCost(fr_name + "_friction", fr_friction, 1e1)

    # Add frame-placement cost
    for fr_name, fr_id in plan_to_model_ids.items():
        # set higher tracking cost on feet
        if 'F' in fr_name:
            w_fr = np.array([10.] * 3 + [0.001] * 3)        # (lin, ang)
        elif 'H' in fr_name:
            w_fr = np.array([2.] * 3 + [0.00001] * 3)
        elif 'R_knee' in fr_name:
            w_fr = np.array([0.5] * 3 + [0.00001] * 3)
        elif 'L_knee' in fr_name:
            w_fr = np.array([2.] * 3 + [0.00001] * 3)
        elif 'torso' in fr_name:
            w_fr = np.array([0.5] * 3 + [0.1] * 3)
            if zero_config is not None:
                w_fr = np.array([1] * 3 + [1.] * 3)
        else:
            raise ValueError(f"Weights to track frame {fr_name} were not set")

        # set the desired frame pose
        fr_Mref = pin.SE3.Identity()
        if fr_name in ee_rpy.keys():
            fr_Mref.rotation = util.util.euler_to_rot(ee_rpy[fr_name])
        fr_Mref.translation = frame_targets_dict[fr_name]

        # add as cost
        activation_fr = crocoddyl.ActivationModelWeightedQuad(w_fr ** 2)
        fr_cost = crocoddyl.CostModelResidual(
            state,
            activation_fr,
            crocoddyl.ResidualModelFramePlacement(state, fr_id, fr_Mref, actuation.nu),
        )
        costs.addCost(fr_name + "_goal", fr_cost, 1e2)

    # Adding state and control regularization terms
    if zero_config is not None:
        x0[3:state.nq] = zero_config[3:]
        x0[-state.nv:] = np.zeros(state.nv)
    w_x = np.array([0.1] * 3 + [10.0] * 3 + [2.] * (state.nv - 6) + [4.] * state.nv)
    activation_xreg = crocoddyl.ActivationModelWeightedQuad(w_x**2)
    x_reg_cost = crocoddyl.CostModelResidual(
        state, activation_xreg, crocoddyl.ResidualModelState(state, x0, actuation.nu)
    )
    u_reg_cost = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    costs.addCost("xReg", x_reg_cost, 5e-3)
    costs.addCost("uReg", u_reg_cost, 1e-6)

    if rcj_constraints is not None:
        raise ValueError("Should not be entering here!")
        # Add the rolling contact joint constraint as cost
        w_rcj = np.array([0.01])        # (lin, ang)
        l_activation_rcj = crocoddyl.ActivationModelWeightedQuad(w_rcj**2)
        l_rcj_residual = ResidualModelStateError(state, 1, nu=actuation.nu)
        l_rcj_residual.constr_ids = [43, 44]  # left and right foot# left side
        l_rcj_cost = crocoddyl.CostModelResidual(
            state,
            l_activation_rcj,
            l_rcj_residual,
        )
        costs.addCost("l_rcj_cost", l_rcj_cost, 1e-2)

        w_rcj = np.array([0.01])        # (lin, ang)
        r_activation_rcj = crocoddyl.ActivationModelWeightedQuad(w_rcj**2)
        r_rcj_residual = ResidualModelStateError(state, 1, nu=actuation.nu)
        r_rcj_residual.constr_ids = [57, 58]  # right and right foot# left side
        r_rcj_cost = crocoddyl.CostModelResidual(
            state,
            r_activation_rcj,
            r_rcj_residual,
        )
        costs.addCost("r_rcj_cost", r_rcj_cost, 1e-2)

    # Adding the state limits penalization
    x_lb = np.concatenate([state.lb[1: state.nv + 1], state.lb[-state.nv:]])
    x_ub = np.concatenate([state.ub[1: state.nv + 1], state.ub[-state.nv:]])
    activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(x_lb, x_ub)
    )
    x_bounds = crocoddyl.CostModelResidual(
        state,
        activation_xbounds,
        crocoddyl.ResidualModelState(state, 0 * x0, actuation.nu),
    )
    costs.addCost("xBounds", x_bounds, 1500.0)

    # Creating the action model
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contacts, costs #, rcj_constraints
    )
    return dmodel


def createSequence(dmodels, DT, N):
    return [
        [crocoddyl.IntegratedActionModelEuler(m, DT)] * N
        + [crocoddyl.IntegratedActionModelEuler(m, 0.0)]
        for m in dmodels
    ]
