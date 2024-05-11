import numpy as np
import crocoddyl
import pinocchio as pin
import util.util


def createDoubleSupportActionModel(state: crocoddyl.StateMultibody,
                                   actuation: crocoddyl.ActuationModelFloatingBase,
                                   rob_data: pin.pinocchio_pywrap.Data,
                                   x0: np.array,
                                   lf_id: int,
                                   rf_id: int,
                                   lh_id: int,
                                   rh_id: int,
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
        np.array([0, 0]),
    )
    rf_contact = crocoddyl.ContactModel6D(
        state,
        rf_id,
        pin.SE3.Identity(),
        pin.LOCAL_WORLD_ALIGNED,
        actuation.nu,
        np.array([0, 0]),
    )
    contacts.addContact("lf_contact", lf_contact)
    contacts.addContact("rf_contact", rf_contact)
    contact_data = contacts.createData(rob_data)

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
    costs.addCost("uReg", u_reg_cost, 1e-4)

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

    # Creating the action model
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contacts, costs
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



def createSequence(dmodels, DT, N):
    return [
        [crocoddyl.IntegratedActionModelEuler(m, DT)] * N
        + [crocoddyl.IntegratedActionModelEuler(m, 0.0)]
        for m in dmodels
    ]
