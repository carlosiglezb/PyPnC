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
        geom_data = {'A1': A, 'b1': b, 'Q': Q, 'U': U}

        # create Casadi vector
        r1 = MX.sym('r1', 3)
        r2 = MX.sym('r2', 3)
        x = vertcat(r1, r2)

        # =========== test point 1 NOT IN COLLISION
        # set initial locations of each simplified rigid body
        r1_val = vertcat(0.0, 0.0, 0.5)
        r2_val = vertcat(0.0, 0.0, 0.0)
        x_val = vertcat(r1_val, r2_val)

        f = SinglePolytopeEllipsoidDistanceCallback('f', geom_data)
        g_dist = Function('g_dist', [x], [f(x)])
        current_alpha = g_dist(x_val)
        expected_alpha = 2.439
        self.assertTrue(current_alpha > 1, "Pair in collision")
        self.assertTrue(np.abs(current_alpha - expected_alpha) < 1e-3, "Error in alpha")

        # -- test Jacobian
        expected_jac = np.array([[0.], [0.], [4.878048], [0.], [0.], [-4.878048]]).T
        J = Function('J', [x], [jacobian(f(x), x)])
        print(f"Jacobian at {x_val} is {J(x_val)}")
        self.assertTrue(np.linalg.norm(expected_jac - J(x_val)) < 1e-3, "Jacobian not correct")

        # -- test Hessian
        expected_hess = np.zeros((6, 6))
        H = Function('H', [x], hessian(f(x), x))
        hess_val, jac_val = H(x_val)
        self.assertTrue(np.linalg.norm(expected_hess - hess_val) < 1e-3, "Hessian not correct")
        self.assertTrue(np.linalg.norm(expected_jac - jac_val.T) < 1e-3, "Jacobian from Hessian not correct")

        # =========== test point 2 NOT IN COLLISION
        x_val = vertcat(0.2, 0., 0.6, 0.0, -0.1, 0.2)
        current_alpha = g_dist(x_val)        # need to run eval again to update dual variables
        expected_alpha = 2.0954
        self.assertTrue(current_alpha > 1, f"Pair not in collision, alpha = {current_alpha}")
        self.assertTrue(np.abs(current_alpha - expected_alpha) < 1e-3, "Error in alpha")

        # -- test Jacobian
        expected_jac = np.array([[4.66849], [0.], [2.904336], [-4.66849], [0.], [-2.904336]]).T
        print(f"Jacobian at {x_val} is {J(x_val)}")
        jac_error = np.linalg.norm(expected_jac - J(x_val))
        self.assertTrue(jac_error < 5e-2, f"Jacobian off by {jac_error}")

        # -- test Hessian
        expected_hess = np.zeros((6, 6))
        H = Function('H', [x], hessian(f(x), x))
        hess_val, jac_val = H(x_val)
        self.assertTrue(np.linalg.norm(expected_hess - hess_val) < 1e-3, "Hessian not correct")
        self.assertTrue(np.linalg.norm(expected_jac - jac_val.T) < 5e-2, "Jacobian from Hessian not correct")

        # =========== test point 3 NOT IN COLLISION
        x_val = vertcat(-0.1, 0., 0.6, 0.0, 0.2, 0.3)
        current_alpha = g_dist(x_val)        # need to run eval again to update dual variables
        expected_alpha = 1.5525
        self.assertTrue(current_alpha > 1, f"Pair not in collision, alpha = {current_alpha}")
        self.assertTrue(np.abs(current_alpha - expected_alpha) < 1e-3, "Error in alpha")

        # -- test Jacobian
        expected_jac = np.array([[0.], [-3.613832], [2.76585], [0.], [3.613832], [-2.76585]]).T
        print(f"Jacobian at {x_val} is {J(x_val)}")
        jac_error = np.linalg.norm(expected_jac - J(x_val))
        self.assertTrue(jac_error < 5e-2, f"Jacobian off by {jac_error}")

        # -- test Hessian
        expected_hess = np.zeros((6, 6))
        H = Function('H', [x], hessian(f(x), x))
        hess_val, jac_val = H(x_val)
        self.assertTrue(np.linalg.norm(expected_hess - hess_val) < 1e-3, "Hessian not correct")
        self.assertTrue(np.linalg.norm(expected_jac - jac_val.T) < 5e-2, "Jacobian from Hessian not correct")

        # =========== test point 4 NOT IN COLLISION
        x_val = vertcat(0., 0., 0.6, 0.1, 0.15, 0.5)
        current_alpha = g_dist(x_val)        # need to run eval again to update dual variables
        expected_alpha = 1.1542
        self.assertTrue(current_alpha > 1, f"Pair not in collision, alpha = {current_alpha}")
        self.assertTrue(np.abs(current_alpha - expected_alpha) < 1e-3, "Error in alpha")

        # -- test Jacobian
        expected_jac = np.array([[-3.5453044], [-5.331045], [0.], [3.5453044], [5.331045], [0.]]).T
        print(f"Jacobian at {x_val} is {J(x_val)}")
        jac_error = np.linalg.norm(expected_jac - J(x_val))
        self.assertTrue(jac_error < 5e-2, f"Jacobian off by {jac_error}")

        # -- test Hessian
        expected_hess = np.zeros((6, 6))
        H = Function('H', [x], hessian(f(x), x))
        hess_val, jac_val = H(x_val)
        self.assertTrue(np.linalg.norm(expected_hess - hess_val) < 1e-3, "Hessian not correct")
        self.assertTrue(np.linalg.norm(expected_jac - jac_val.T) < 5e-2, "Jacobian from Hessian not correct")


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

        f = SinglePolytopePolytopeDistanceCallback('f', geom_data)
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
        f = SinglePolytopePolytopeDistanceCallback('f', geom_data)
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


    def test_min_distance_polytope_ellipsoid_dcol_2points(self):
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
        Q = np.eye(3)
        radius = 0.03
        U = (1/radius) * np.eye(3)       # Cholesky factorization of end effector's sphere radius
        geom_data = {'A1': A1, 'b1': b1, 'Q': Q, 'U': U}
        expected_z_distance = 0.175+radius

        # optimization problem formulation
        p1 = MX.sym('p1', 3)
        p2 = MX.sym('p2', 3)
        x = vertcat(p1, p2)
        f = SinglePolytopeEllipsoidDistanceCallback('f', geom_data)
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
                "derivative_test_tol": 0.005}
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
            else:
                self.assertTrue(False, "Points were not mostly above/below each other")


    def test_min_distance_capsule_ellipsoid_dcol_2points(self):
        R = 0.11
        L = 0.32
        Q = np.eye(3)
        sphere_radius = 0.08
        U = (1/sphere_radius) * np.eye(3)       # Cholesky factorization of end effector's sphere radius

        x_init = np.array([[0.0], [0.0], [L/2 + R + sphere_radius], [0.0], [0.0], [0.0]])

        x, alpha, dual_v = solve_capsule_ellipsoid_min_prox(R, L, Q, U, x_init[:3], x_init[3:])

        self.assertAlmostEqual()

if __name__ == '__main__':
    unittest.main()
