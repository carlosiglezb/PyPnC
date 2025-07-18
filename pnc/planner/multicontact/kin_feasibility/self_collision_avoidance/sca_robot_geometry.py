import pinocchio as pin
import numpy as np

class SCARobotGeometry:
    def __init__(self, robot_model_path: str,
                 urdf_path: str,
                 plan_to_model_frames: dict[str: str]):
        # Load the robot and geometry model and collision models from urdf
        robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path, robot_model_path, root_joint=pin.JointModelFreeFlyer())
        geom_model = pin.buildGeomFromUrdf(robot.model,
                                           urdf_path,
                                           pin.GeometryType.COLLISION)
        geom_model.addAllCollisionPairs()

        sca_link_names = plan_to_model_frames.values()

        # extract description of primitives for each link in the sca_link_names
        self._geometry_primitives = {}
        for lnk in sca_link_names:
            for gm in geom_model.geometryObjects:
                # TODO figure out what to do with cascaded collision bodies
                # This will currently only save the settings of the last collision item
                if lnk in gm.name:
                    halfspace_params = {}
                    if gm.meshPath == 'BOX':
                        box_half_side = gm.geometry.halfSide.reshape(-1, 1)
                        halfspace_params['A'] = np.vstack((np.eye(3), -np.eye(3)))
                        halfspace_params['b'] = np.vstack((box_half_side, box_half_side))
                    elif gm.meshPath == 'SPHERE':
                        sphere_radius = gm.geometry.radius
                        halfspace_params['U'] = 1./sphere_radius * np.eye(3)
                    else:
                        raise NotImplementedError("Only box primitives are currently supported")
                    self._geometry_primitives[lnk] = halfspace_params

        self.geom_model = geom_model
        self._plan_to_model_frames = plan_to_model_frames

    def get_box_representation(self, link_name: str) -> dict[str, np.array]:
        return self._geometry_primitives[self._plan_to_model_frames.get(link_name)]

    def get_sphere_representation(self, link_name: str) -> dict[str, np.array]:
        return self._geometry_primitives[self._plan_to_model_frames.get(link_name)]

    def get_primitive_shape_type(self, link_name: str) -> str:
        if 'A' in self._geometry_primitives[self._plan_to_model_frames.get(link_name)]:
            return 'box'
        elif 'U' in self._geometry_primitives[self._plan_to_model_frames.get(link_name)]:
            return 'sphere'
        return 'unknown'


    def is_link_in_sca_list(self, link_name) -> bool:
        return self._plan_to_model_frames[link_name] in self._geometry_primitives.keys()
