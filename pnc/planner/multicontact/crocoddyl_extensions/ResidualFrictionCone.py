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
        # F_local = self.plane_rotation.T @ F
        # data.r[0] = self.mu * F_local[2] - np.sqrt(F_local[0] ** 2 + F_local[1] ** 2)

        # TODO delete after checking above is right
        # if 'LH' in self.contact_name:
        #     data.r[0] = -self.mu * F[1] - np.sqrt(F[0] ** 2 + F[2] ** 2)
        # elif 'RH' in self.contact_name:
        #     data.r[0] = self.mu * F[1] - np.sqrt(F[0] ** 2 + F[2] ** 2)
        # else:
        #     data.r[0] = self.mu * F[2] - np.sqrt(F[0] ** 2 + F[1] ** 2)
        data.r[0] = self.mu * F[2] - np.sqrt(F[0] ** 2 + F[1] ** 2)

    def calcDiff(self, data, x, u=None):
        F = data.shared.contacts.contacts[self.contact_name].f.vector[:3]

        # FIXME testing w/planes
        # F_local = self.plane_rotation.T @ F
        # self.dcone_df[0, 0] = -F_local[0] / np.sqrt(F_local[0] ** 2 + F_local[1] ** 2)
        # self.dcone_df[0, 1] = -F_local[1] / np.sqrt(F_local[0] ** 2 + F_local[1] ** 2)
        # self.dcone_df[0, 2] = self.mu

        # TODO apply rotation matrix rather than by name
        # if 'LH' in self.contact_name:
        #     self.dcone_df[0, 0] = -F[0] / np.sqrt(F[0] ** 2 + F[2] ** 2)
        #     self.dcone_df[0, 1] = -self.mu
        #     self.dcone_df[0, 2] = -F[2] / np.sqrt(F[0] ** 2 + F[2] ** 2)
        # elif 'RH' in self.contact_name:
        #     self.dcone_df[0, 0] = -F[0] / np.sqrt(F[0] ** 2 + F[2] ** 2)
        #     self.dcone_df[0, 1] = self.mu
        #     self.dcone_df[0, 2] = -F[2] / np.sqrt(F[0] ** 2 + F[2] ** 2)
        # else:
        #     self.dcone_df[0, 0] = -F[0] / np.sqrt(F[0] ** 2 + F[1] ** 2)
        #     self.dcone_df[0, 1] = -F[1] / np.sqrt(F[0] ** 2 + F[1] ** 2)
        #     self.dcone_df[0, 2] = self.mu
        #
        self.dcone_df[0, 0] = -F[0] / np.sqrt(F[0] ** 2 + F[1] ** 2)
        self.dcone_df[0, 1] = -F[1] / np.sqrt(F[0] ** 2 + F[1] ** 2)
        self.dcone_df[0, 2] = self.mu
        self.df_dx = data.shared.contacts.contacts[self.contact_name].df_dx[:3]
        self.df_du = data.shared.contacts.contacts[self.contact_name].df_du[:3]

        data.Rx = self.dcone_df @ self.df_dx
        data.Ru = self.dcone_df @ self.df_du