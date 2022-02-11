import unittest
from QFunctionNN import QFunctionNN
import gym

env = gym.make('CubeCrash-v0')



class MyTestCase(unittest.TestCase):
    def test_QF_conv_out(self):
        a = QFunctionNN(lr=0.9995, state_shape=env.observation_space.shape, actions_n=env.action_space.n)
        conv_out = a.size_conv_out()
        print(conv_out)

if __name__ == '__main__':
    unittest.main()
