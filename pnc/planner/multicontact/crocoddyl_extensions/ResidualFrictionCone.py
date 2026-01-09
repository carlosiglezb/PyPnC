"""
License: BSD 3-Clause License
Copyright (C) 2023, New York University

Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import numpy as np
import crocoddyl


class ResidualFrictionCone(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, contact_name, mu, nu, plane_rotation=None):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 1, nu, True, True, True)

        if plane_rotation is None:
            self.plane_rotation = np.eye(3)
        else:
            self.plane_rotation = plane_rotation
        self.mu = mu
        self.contact_name = contact_name

        self.dcone_df = np.zeros((1, 3))
        self.df_dx = np.zeros((3, self.state.ndx))
        self.df_du = np.zeros((3, self.nu))

    def calc(self, data, x, u=None):
        F = data.shared.contacts.contacts[self.contact_name].f.vector[:3]

        # FIXME testing w/planes
        F_local = self.plane_rotation.T @ F
        data.r[0] = self.mu * F_local[2] - np.sqrt(F_local[0] ** 2 + F_local[1] ** 2 + 1e-3)

    def calcDiff(self, data, x, u=None):
        F = data.shared.contacts.contacts[self.contact_name].f.vector[:3]

        # FIXME testing w/planes
        F_local = self.plane_rotation.T @ F
        self.dcone_df[0, 0] = -F_local[0] / np.sqrt(F_local[0] ** 2 + F_local[1] ** 2)
        self.dcone_df[0, 1] = -F_local[1] / np.sqrt(F_local[0] ** 2 + F_local[1] ** 2)
        self.dcone_df[0, 2] = self.mu
        # chain rule:dcone/dF_world = dcone/dF_local * R^T
        self.dcone_df = self.dcone_df @ self.plane_rotation.T

        self.df_dx = data.shared.contacts.contacts[self.contact_name].df_dx[:3]
        self.df_du = data.shared.contacts.contacts[self.contact_name].df_du[:3]

        data.Rx = self.dcone_df @ self.df_dx
        data.Ru = self.dcone_df @ self.df_du


class ResidualLinearizedFrictionCone(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, contact_name, mu, nu, plane_rotation=None):
        # We now have 4 inequality constraints (represented as residuals)
        crocoddyl.ResidualModelAbstract.__init__(self, state, 4, nu, True, True, True)

        self.mu = mu
        self.contact_name = contact_name
        self.plane_rotation = np.eye(3) if plane_rotation is None else plane_rotation

        # Pre-allocate matrices for chain rule
        self.df_dx = np.zeros((3, self.state.ndx))
        self.df_du = np.zeros((3, self.nu))

    def calc(self, data, x, u=None):
        # 1. Get world force and rotate to local frame
        F_world = data.shared.contacts.contacts[self.contact_name].f.vector[:3]
        F = self.plane_rotation.T @ F_world

        # 2. Linearized constraints: mu*Fz +/- Fx >= 0 and mu*Fz +/- Fy >= 0
        # We write them as: r = [Fx - mu*Fz, -Fx - mu*Fz, Fy - mu*Fz, -Fy - mu*Fz]
        # In optimization, we usually want r <= 0
        data.r[0] = F[0] - self.mu * F[2]
        data.r[1] = -F[0] - self.mu * F[2]
        data.r[2] = F[1] - self.mu * F[2]
        data.r[3] = -F[1] - self.mu * F[2]

    def calcDiff(self, data, x, u=None):
        # Jacobian of the cone w.r.t local force F
        # Row 0: d(Fx - mu*Fz)/dF = [1, 0, -mu]
        # Row 1: d(-Fx - mu*Fz)/dF = [-1, 0, -mu] ... etc.
        dcone_df_local = np.array([
            [1, 0, -self.mu],
            [-1, 0, -self.mu],
            [0, 1, -self.mu],
            [0, -1, -self.mu]
        ])

        # Chain rule: dcone/dF_world = dcone/dF_local * R^T
        dcone_df_world = dcone_df_local @ self.plane_rotation.T

        self.df_dx = data.shared.contacts.contacts[self.contact_name].df_dx[:3]
        self.df_du = data.shared.contacts.contacts[self.contact_name].df_du[:3]

        data.Rx = dcone_df_world @ self.df_dx
        data.Ru = dcone_df_world @ self.df_du