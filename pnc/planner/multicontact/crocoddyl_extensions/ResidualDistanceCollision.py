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


class ResidualDataDistanceCollision(ResidualDataAbstract):
    def __init__(self, model, data):
        super().__init__(model, data)
        self.r = np.zeros(1)
        self.req = hppfcl.DistanceRequest()
        self.res = hppfcl.DistanceResult()

class ResidualDistanceCollision(ResidualModelAbstract):
    def __init__(self,
                 state: StateMultibody,
                 nu: int,
                 geom_model: pin.GeometryModel,
                 pair_id: int,
                 args=None, kwargs=None):
        # We assume Base is a class like ResidualAbstract
        super().__init__(state, 1, nu, True, False, False)
        self.geometry = geom_model
        self.pair_id_ = pair_id
        self.nearestPoint1 = np.zeros(3)
        self.nearestPoint2 = np.zeros(3)
        self.normal = np.zeros(3)

        self.pin_model_ = state.pinocchio
        self.nv_ = self.pin_model_.nv

        if pair_id >= len(geom_model.collisionPairs):
            raise ValueError(
                "Invalid argument: the pair index is wrong (it does not exist in the geometry model)"
            )

    def copy(self, ResidualDistanceCollision, *args, **kwargs):
        return self.copy(*args, **kwargs)

    def createData(self, data_collector: DataCollectorAbstract) -> ResidualDataDistanceCollision:
        # Create and return an instance of our custom data class
        return ResidualDataDistanceCollision(self, data_collector)

    def calc(self, data: ResidualModelAbstract, x: np.ndarray, u: Optional[np.ndarray] = None):

        # compute the distance for the collision pair pair_id_
        cp = self.geometry.collisionPairs[self.pair_id_]
        geom_1 = self.geometry.geometryObjects[cp.first]
        geom_2 = self.geometry.geometryObjects[cp.second]
        joint_id_1 = geom_1.parentJoint
        joint_id_2 = geom_2.parentJoint

        # pinocchio.Data object is inside the data class
        pin_data = data.shared.pinocchio

        # get oMg for both geometries
        if joint_id_1 > 0:
            M1 = pin_data.oMi[joint_id_1] * geom_1.placement
        else:
            M1 = geom_1.placement

        if joint_id_2 > 0:
            M2 = pin_data.oMi[joint_id_2] * geom_2.placement
        else:
            M2 = geom_2.placement

        # Convert Pinocchio SE3 to FCL Transform
        T1 = hppfcl.Transform3f(M1.rotation, M1.translation)
        T2 = hppfcl.Transform3f(M2.rotation, M2.translation)
        data.res.clear()
        dist = hppfcl.distance(geom_1.geometry, T1, geom_2.geometry, T2, data.req, data.res)
        data.r[0] = dist


    def calcDiff(self, data: ResidualModelAbstract, x: np.ndarray, u: Optional[np.ndarray] = None):
        # type hinting for clarity
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
        cp1 = data.res.getNearestPoint1()
        cp2 = data.res.getNearestPoint2()
        data.cp1 = cp1
        data.cp2 = cp2

        # Vector from frame 1 center to p1
        f1p1 = cp1 - data.shared.pinocchio.oMf[geom_1.parentFrame].translation
        f1Mp1 = pin.SE3.Identity()
        f1Mp1.translation = f1p1

        # Transport the jacobian of frame 1 into the jacobian associated to cp1
        J1 = f1Mp1.toActionMatrixInverse() @ J1

        # Vector from frame 2 center to p2
        f2p2 = cp2 - data.shared.pinocchio.oMf[geom_2.parentFrame].translation
        f2Mp2 = pin.SE3.Identity()
        f2Mp2.translation = f2p2

        # Transport the jacobian of frame 2 into the jacobian associated to cp2
        J2 = f2Mp2.toActionMatrixInverse() @ J2

        # calculate the Jacobian
        data.Rx[:nv] = -data.res.normal.reshape(-1, 1).T @ (J1[:3, :] - J2[:3, :])

    @property
    def pair_id(self) -> int:
        return self.pair_id_

    @pair_id.setter
    def pair_id(self, pair_id: int):
        self.pair_id_ = pair_id
