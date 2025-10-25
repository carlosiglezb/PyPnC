import pinocchio as pin
import hppfcl
import numpy as np
from typing import Optional
from crocoddyl.libcrocoddyl_pywrap import *

"""
Residual model for distance between two collision objects
Implementation follows the derivation in package colmpc:

Haffemayer, A., Jordana, A., Fourmy, M., Wojciechowski, K., Saurel, G., 
Petrík, V., Lamiraux, F. and Mansard, N., 2024, June. Model predictive 
control under hard collision avoidance constraints for a robotic arm. 
In 2024 21st International Conference on Ubiquitous Robots (UR) 
(pp. 701-706). IEEE.
"""

def to_fcl_transform3f(T: pin.SE3):
    return hppfcl.Transform3f(T.rotation, T.translation)


class ResidualDataDistanceCollision(ResidualDataAbstract):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.r = np.zeros(1)
        # self.J1 = np.zeros((6, model.nv))
        # self.J2 = np.zeros((6, model.nv))
        # self.oMg_id_1 = pin.SE3.Identity()
        # self.oMg_id_2 = pin.SE3.Identity()
        # self.cp1 = np.zeros(3)
        # self.cp2 = np.zeros(3)

    def copy(self, ResidualDataDistanceCollision, *args, **kwargs):
        return self.copy(*args, **kwargs)

    def __copy__(self, ResidualDataDistanceCollision, *args, **kwargs):
        raise NotImplementedError("[ResidualDataDistanceCollision] Copy is not implemented in Python")

    def __deepcopy__(self, ResidualDataDistanceCollision, *args, **kwargs):
        raise NotImplementedError("[ResidualDataDistanceCollision] Deep copy method is not implemented in Python")

    def __reduce__(self, p_object, *args, **kwargs):
        raise NotImplementedError("[ResidualDataDistanceCollision] Reduce is not implemented in Python")

class ResidualDistanceCollision(ResidualModelAbstract):
    def __init__(self, state: StateMultibody, nu: int, geom_model: pin.GeometryModel, pair_id: int, args, kwargs):
        # We assume Base is a class like ResidualAbstract
        super().__init__(state, 1, nu, True, False, False)
        # self.state = state
        # self.nu = nu
        self.geometry = geom_model
        self.pair_id_ = pair_id

        self.pin_model_ = state.pinocchio
        self.nv_ = self.pin_model_.nv
        hppfcl.DistanceResult.__init__(self)
        self.hppfcl_dreq = hppfcl.DistanceRequest()
        self.hppfcl_dres = hppfcl.DistanceResult()
        self.hppfcl_creq = hppfcl.CollisionRequest()
        self.hppfcl_cres = hppfcl.CollisionResult()

        if pair_id >= len(geom_model.collisionPairs):
            raise ValueError(
                "Invalid argument: the pair index is wrong (it does not exist in the geometry model)"
            )

    def copy(self, ResidualDistanceCollision, *args, **kwargs):
        return self.copy(*args, **kwargs)

    # def createData(self, data_collector: DataCollectorAbstract) -> ResidualDataDistanceCollision:
    #     # This will create and return an instance of our custom data class
    #     # It assumes `pin_model_` is accessible here, which it is since it's a member variable
    #     return ResidualDataDistanceCollision(self.pin_model_, data_collector)
    #     # return ResidualDataDistanceCollision(pin_model, *args, **kwargs)

    def calc(self, data: ResidualModelAbstract, x: np.ndarray, u: Optional[np.ndarray] = None):
        # Using type hinting to replace static_cast<Data*>
        # clear the hppfcl results
        # data.res.clear()

        # computes the distance for the collision pair pair_id_
        cp = self.geometry.collisionPairs[self.pair_id_]
        geom_1 = self.geometry.geometryObjects[cp.first]
        geom_2 = self.geometry.geometryObjects[cp.second]
        joint_id_1 = geom_1.parentJoint
        joint_id_2 = geom_2.parentJoint

        # pinocchio.Data object is now inside the custom data class
        pin_data = data.shared.pinocchio
        # pin_data = data

        # get oMg for both geometries
        if joint_id_1 > 0:
            data.oMg_id_1 = pin_data.oMi[joint_id_1] * geom_1.placement
        else:
            data.oMg_id_1 = geom_1.placement

        if joint_id_2 > 0:
            data.oMg_id_2 = pin_data.oMi[joint_id_2] * geom_2.placement
        else:
            data.oMg_id_2 = geom_2.placement

        # hppfcl distance calculation
        # .geometry is assumed to be an HPP-FCL object
        data.r[0] = hppfcl.distance(
            geom_1.geometry, to_fcl_transform3f(data.oMg_id_1),
            geom_2.geometry, to_fcl_transform3f(data.oMg_id_2),
            self.hppfcl_dreq, self.hppfcl_dres
        )

        # check if in collision to use as signed distance
        self.hppfcl_creq = hppfcl.CollisionRequest()
        self.hppfcl_cres = hppfcl.CollisionResult()
        hppfcl.collide(geom_1.geometry, to_fcl_transform3f(data.oMg_id_1), geom_2.geometry,
                       to_fcl_transform3f(data.oMg_id_2),
                       self.hppfcl_creq, self.hppfcl_cres)
        if self.hppfcl_cres.isCollision():
            data.r[0] = -data.r[0]

    def calcDiff(self, data: ResidualModelAbstract, x: np.ndarray, u: Optional[np.ndarray] = None):
        # Using type hinting for clarity
        nv = self.state.nv

        cp = self.geometry.collisionPairs[self.pair_id_]
        geom_1 = self.geometry.geometryObjects[cp.first]
        geom_2 = self.geometry.geometryObjects[cp.second]

        # In Pinocchio Python bindings, getFrameJacobian is called directly
        J1 = pin.getFrameJacobian(self.pin_model_, data.shared.pinocchio, geom_1.parentFrame,
                             pin.LOCAL_WORLD_ALIGNED)

        J2 = pin.getFrameJacobian(self.pin_model_, data.shared.pinocchio, geom_2.parentFrame,
                             pin.LOCAL_WORLD_ALIGNED)

        # Getting the nearest points belonging to the collision shapes
        cp1 = self.hppfcl_dres.getNearestPoint1()
        cp2 = self.hppfcl_dres.getNearestPoint2()
        # data.cp1 = cp1
        # data.cp2 = cp2

        # Vector from frame 1 center to p1
        f1p1 = cp1 - data.shared.pinocchio.oMf[geom_1.parentFrame].translation
        f1Mp1 = pin.SE3.Identity()
        f1Mp1.translation = f1p1

        # Transport the jacobian of frame 1 into the jacobian associated to cp1
        # In Pinocchio Python, .toActionMatrixInverse() is a method
        J1 = f1Mp1.toActionMatrixInverse() @ J1

        # Vector from frame 2 center to p2
        f2p2 = cp2 - data.shared.pinocchio.oMf[geom_2.parentFrame].translation
        f2Mp2 = pin.SE3.Identity()
        f2Mp2.translation = f2p2

        # Transport the jacobian of frame 2 into the jacobian associated to cp2
        J2 = f2Mp2.toActionMatrixInverse() @ J2

        # calculate the Jacobian, assuming data.Rx is a numpy array
        # -d->res.normal.transpose() becomes -data.res.normal.T
        # topRows<3>() becomes a slice [0:3, :]
        data.Rx[:nv] = -self.hppfcl_dres.normal.reshape(-1, 1).T @ (J1[:3, :] - J2[:3, :])
        if self.hppfcl_cres.isCollision():
            data.Rx[:nv] = -data.Rx[:nv]

    @property
    def pair_id(self) -> int:
        return self.pair_id_

    @pair_id.setter
    def pair_id(self, pair_id: int):
        self.pair_id_ = pair_id
