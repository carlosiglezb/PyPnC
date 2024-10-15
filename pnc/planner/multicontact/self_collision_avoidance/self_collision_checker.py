import pinocchio as pin

class SelfCollisionChecker:
    def __init__(self, robot_model_path, urdf_path, srdf_path):
        # Load the robot and geometry model with collisions from srdf
        robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path, robot_model_path, root_joint=pin.JointModelFreeFlyer())
        geom_model = pin.buildGeomFromUrdf(robot.model,
                                           urdf_path,
                                           robot_model_path,
                                           pin.GeometryType.COLLISION)
        geom_model.addAllCollisionPairs()
        pin.removeCollisionPairs(robot.model, geom_model, srdf_path)

        self.geom_data = pin.GeometryData(geom_model)
        self.robot = robot
        self.geom_model = geom_model

    def get_rob_geom_models(self):
        return self.robot, self.geom_model

    def check_collisions(self, q):
        b_collision_detected = False

        geom_model = self.geom_model
        geom_data = self.geom_data

        # Compute all the collisions
        pin.computeCollisions(
            self.robot.model,
            self.robot.data,
            geom_model,
            geom_data,
            q,
            False,
        )

        # Check for collisions at each time step
        for k in range(len(geom_model.collisionPairs)):
            cr = geom_data.collisionResults[k]
            cp = geom_model.collisionPairs[k]
            if cr.isCollision():
                print("Collision between:",
                geom_model.geometryObjects[cp.first].name, ",",
                geom_model.geometryObjects[cp.second].name)
                b_collision_detected = True
                break

        return b_collision_detected