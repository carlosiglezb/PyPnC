import unittest

import numpy as np
from pnc.planner.multicontact.kin_feasibility.fastpathplanning.fastpathplanning import unassigned_box_seq_interpolator


class TestFPPTools(unittest.TestCase):
    def test_unassigned_box_seq_interp_uneven(self):
        test_frame = 'torso'

        # generate sequence with uneven number of undefined boxes
        des_box_seq = [{test_frame: [0]},
                       {test_frame: [1]},
                       {test_frame: [1]}]
        box_seq = [{test_frame: [0]},
                   {test_frame: [np.nan]},
                   {test_frame: [np.nan]}]
        last_box_seq = {test_frame: [0, 1, 2]}

        # call method
        unassigned_box_seq_interpolator(box_seq, last_box_seq, test_frame)

        # compare against expected result
        for bs, des_bs in zip(box_seq, des_box_seq):
            for box, des_box in zip(bs[test_frame], des_bs[test_frame]):
                self.assertEqual(box, des_box)


if __name__ == '__main__':
    unittest.main()
