import unittest
from DuelingDQLNN import QFunctionNN
import gym
import torch
from utils import ExperienceBuffer
import numpy as np

env = gym.make('CubeCrash-v0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = QFunctionNN(lr=0.9995, state_shape=env.observation_space.shape, actions_n=env.action_space.n)
net = QFunctionNN(lr=0.9995, state_shape=(3, 84, 84), actions_n=env.action_space.n, name="net_v0",
                  checkpoint_dir="./saved_models")
zrs = torch.zeros([1, 3, 84, 84]).to(device)


class QFTestcase(unittest.TestCase):
    def test_QF_conv_out(self):
        conv_out = net.conv(zrs)
        self.assertEqual(type(conv_out), torch.Tensor)

    def test_QF_conv_out_size(self):
        conv_out_shape = net.get_fc_input_size()
        print(conv_out_shape)
        self.assertEqual(conv_out_shape, 3136)

    def test_QF_forward(self):
        out = net.forward(zrs)
        self.assertEqual(type(out), torch.Tensor)
        self.assertEqual(int(out.size()[0]), net.actions_n)

    def test_save(self):
        net.zero_grad()
        out1 = net.forward(zrs)
        out2 = net.forward(zrs+1)
        loss = net.loss(out1, out2)
        loss.backward()
        net.optimizer.step()
        net.save_checkpoint()


class expBufferTestcase(unittest.TestCase):
    def test_constructor(self):
        exp_buffer = ExperienceBuffer(max_size=5, input_shape=[4, 84, 84])
        self.assertEqual(type(exp_buffer), ExperienceBuffer)

    def test_insert_buffer(self):
        exp_buffer = ExperienceBuffer(max_size=5, input_shape=[4, 84, 84])
        exp_buffer.insert(1, 2, 3, 2, False)
        exp_buffer.insert(1, 3, 45, 6, False)
        self.assertEqual(exp_buffer.mem_idx, 2)

    def test_max_len(self):
        exp_buffer = ExperienceBuffer(max_size=5, input_shape=[4, 84, 84])
        exp_buffer.insert(1, 2, 3, 2, False)
        exp_buffer.insert(1, 3, 45, 6, True)
        exp_buffer.insert(3, 3, 5, 6, False)
        exp_buffer.insert(3, 5, 5, 6, True)
        exp_buffer.insert(4, 3, 5, 6, False)
        s = min(exp_buffer.mem_idx, exp_buffer.mem_size)
        self.assertEqual(s, 5)

        exp_buffer.insert(3, 5, 5, 6, False)
        self.assertEqual(s, 5)

    def teste_sample(self):
        exp_buffer = ExperienceBuffer(max_size=5, input_shape=[4, 84, 84])
        exp_buffer.insert(1, 2, 3, 2, False)
        exp_buffer.insert(1, 3, 45, 6, False)
        exp_buffer.insert(3, 3, 5, 6, False)
        exp_buffer.insert(3, 5, 5, 6, True)
        exp_buffer.insert(4, 3, 5, 6, False)
        states, actions, rewards, states_, dones = exp_buffer.sample(3)
        self.assertEqual(dones.size, 3)


if __name__ == '__main__':
    unittest.main()
