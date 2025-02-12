import unittest
import os, sys
import casadi

cwd = os.getcwd()
sys.path.append(cwd)

from pnc.planner.multicontact.kin_feasibility.casadi_ocp_constraints.casadi_ocp_functions import *

class TestCasadiOcpCallbacks(unittest.TestCase):
    def test_dcol_min_distance_callback(self):
        # # G1 robot paths
        # robot_model_path = cwd + "/robot_model/g1_description/"
        # urdf_path = robot_model_path + "g1_cube_points_collisions.urdf"
        # srdf_path = robot_model_path + "config/g1_cube_points.srdf"
        #
        # # create robot model and geom model from srdf
        # robot = pin.RobotWrapper.BuildFromURDF(
        #     urdf_path, robot_model_path, root_joint=pin.JointModelFreeFlyer())
        # geom_model = pin.buildGeomFromUrdf(robot.model,
        #                                    urdf_path,
        #                                    robot_model_path,
        #                                    pin.GeometryType.COLLISION)
        # geom_model.addAllCollisionPairs()
        # pin.removeCollisionPairs(robot.model, geom_model, srdf_path)
        #
        # q_test = copy.copy(robot.q0)

        # TODO get A, b, Q, r1, r2 from robot model (URDF)
        A = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]
                      ])
        b = np.array([[0.07],
                     [0.105],
                     [0.175],
                     [0.07],
                     [0.105],
                     [0.175]]
                     )
        Q = np.eye(3)
        radius = 0.03
        U = (1/radius) * np.eye(3)       # Cholesky factorization of end effector's sphere radius
        geom_data = {'A': A, 'b': b, 'Q': Q, 'U': U}

        # create Casadi vector
        r1 = MX.sym('r1', 3)
        r2 = MX.sym('r2', 3)

        # set initial locations of each simplified rigid body
        r1_val = vertcat(0.0, 0.0, 0.5)
        r2_val = vertcat(0.0, 0.0, 0.0)
        actual_distance = 0.5 - 0.175 - radius

        f = DColMinDistancePairsCallback('f', geom_data)
        g_dist = Function('g_dist', [r1, r2], [f(r1, r2)])
        current_distance = g_dist(r1_val, r2_val)
        self.assertTrue(current_distance > 0, "Pair in collision")  # add assertion here
        self.assertTrue(np.linalg.norm(current_distance - (actual_distance)) < 1e-6, "Distance not correct")

    def test_dcol_min_poly_distance_callback(self):
        A1 = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]
                      ])
        b1 = np.array([[0.07],
                     [0.105],
                     [0.175],
                     [0.07],
                     [0.105],
                     [0.175]]
                     )
        A2 = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]
                      ])
        radius = 0.03
        b2 = np.array([[radius],
                     [radius],
                     [radius],
                     [radius],
                     [radius],
                     [radius]]
                     )
        Q = np.eye(3)
        geom_data = {'A1': A1, 'b1': b1, 'A2': A2, 'b2': b2, 'Q': Q}

        # create Casadi vector
        r1 = MX.sym('r1', 3)
        r2 = MX.sym('r2', 3)
        x = vertcat(r1,r2)

        # set initial locations of each simplified rigid body
        r1_val = vertcat(0., 0., 0.5)
        r2_val = vertcat(0., 0., 0.0)

        f = DColMinSinglePolytopesDistanceCallback('f', geom_data)
        g_dist = Function('g_dist', [x], [f(x)])

        # =========== test point 1 NOT in collision
        x_val = vertcat(r1_val, r2_val)
        current_distance = g_dist(x_val)
        self.assertTrue(current_distance > 1, "Pair in collision")

        # -- test Jacobian
        expected_jac = np.array([[0.], [0.], [4.87805], [0.], [0.], [-4.87805]]).T
        J = Function('J', [x], [jacobian(f(x), x)])
        self.assertTrue(np.linalg.norm(expected_jac - J(x_val)) < 1e-3, "Jacobian not correct")

        # -- test Hessian
        expected_hess = np.zeros((6,6))
        H = Function('H', [x], hessian(f(x), x))
        hess_val, jac_val = H(x_val)
        self.assertTrue(np.linalg.norm(expected_hess - hess_val) < 1e-3, "Hessian not correct")
        self.assertTrue(np.linalg.norm(expected_jac - jac_val.T) < 1e-3, "Jacobian from Hessian not correct")

        # =========== test point 2 IN COLLISION
        x_val = vertcat(0.2, 0., 0.6, 0.2, -0.1, 0.6)
        current_distance = g_dist(x_val)        # need to run eval again to update dual variables
        self.assertTrue(current_distance < 1, "Pair not in collision")  # add assertion here

        # -- test Jacobian
        expected_jac = np.array([[0.], [7.4074], [0.], [0.], [-7.4074], [0.]]).T
        self.assertTrue(np.linalg.norm(expected_jac - J(x_val)) < 1e-3, "Jacobian not correct")

        # -- test Hessian
        expected_hess = np.zeros((6,6))
        H = Function('H', [x], hessian(f(x), x))
        hess_val, jac_val = H(x_val)
        self.assertTrue(np.linalg.norm(expected_hess - hess_val) < 1e-3, "Hessian not correct")
        self.assertTrue(np.linalg.norm(expected_jac - jac_val.T) < 1e-3, "Jacobian from Hessian not correct")


    def test_min_distance_polytopes_dcol_2points(self):
        N_tests = 10
        A1 = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]
                      ])
        b1 = np.array([[0.07],
                     [0.105],
                     [0.175],
                     [0.07],
                     [0.105],
                     [0.175]]
                     )
        A2 = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]
                      ])
        radius = 0.03
        b2 = np.array([[radius],
                     [radius],
                     [radius],
                     [radius],
                     [radius],
                     [radius]]
                     )
        Q = np.eye(3)
        geom_data = {'A1': A1, 'b1': b1, 'A2': A2, 'b2': b2, 'Q': Q}
        expected_z_distance = 0.175+radius

        # optimization problem formulation
        p1 = MX.sym('p1', 3)
        p2 = MX.sym('p2', 3)
        x = vertcat(p1, p2)
        f = DColMinSinglePolytopesDistanceCallback('f', geom_data)
        dist_p1_p2 = norm_2(x[:3] - x[3:])     # (x,y)-distance between two points
        nlp = {'x': x,
               'f': dist_p1_p2,
               'g': f(x)
               }
        opts = {
            "ipopt": {
                "hessian_approximation": "exact",  # limited-memory
                "max_iter": 100,
                "derivative_test": "second-order",
                "derivative_test_print_all": "no",
                "derivative_test_perturbation": 1e-6,
                "derivative_test_tol": 0.001}
        }
        solver = nlpsol('solver', 'ipopt', nlp, opts)

        x_init = np.array([0.0, 0.0, 2.25, 0.0, 0.0, 0.0])
        for i in range(N_tests):
            x_init[:2] = 0.2*np.random.random(2)
            x_init[2] = 2.25 + 0.4*np.random.random()
            print(f"Test points {i}: p1 = {x_init[:3]}, p2 = {x_init[3:]}")
            sol = solver(x0=x_init,lbg=1.0, ubg=casadi.inf)

            sol_stats = solver.stats()
            self.assertTrue(sol_stats['success'], "Optimization failed")

            # check if the solution seems correct
            x_sol = sol['x'].full()
            if x_init[2] > x_init[-1]:  # if initialized with p_1 above p2
                computed_z_dist = np.linalg.norm(x_sol[:3] - x_sol[3:])
                dist_z_error = computed_z_dist - expected_z_distance
                self.assertTrue(dist_z_error < 1e-3, f"Min. Distance between points is {dist_z_error}")
                self.assertTrue(dist_z_error + 0.01 > 1e-3, f"Min. Distance between points is {dist_z_error}")
            else:
                self.assertTrue(False, "Points were not mostly above/below each other")

if __name__ == '__main__':
    unittest.main()
