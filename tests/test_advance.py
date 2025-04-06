from aleatory.processes import CEVProcess
import unittest
import numpy as np


class TestAdvance(unittest.TestCase):
    def test_advance_method(self):
        cev = CEVProcess()

        cev.start(np.zeros(5))
        for t in np.arange(0.1, 1.01, 0.1):
            cev.advance(t)
            print(cev.val)


if __name__ == "__main__":
    unittest.main()

"""
Output:

[0.07483507  0.06892834  0.0103775   0.08222023 -0.00520991]
[0.1242685  0.09343143 0.0547229  0.04053699 0.07427636]
[0.19546715 0.1776751  0.09312892 0.07602916 0.09573145]
[0.2762749  0.25656871 0.17613864 0.12536773 0.09022976]
[0.29868798 0.33389956 0.28908409 0.15776951 0.09993488]
[0.40701358 0.45013018 0.30473939 0.23695351 0.16412336]
[0.47892648 0.54385911 0.3692668  0.29175553 0.20496346]
[0.57069992 0.66287047 0.40147713 0.33604199 0.20004954]
[0.58912079 0.73014259 0.45835246 0.37663655 0.22481997]
[0.63629761 0.7450066  0.50876839 0.42058831 0.32175386]

"""