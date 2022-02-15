import torch
import numpy as np
from QFunctionNN import QFunctionNN
from utils import ExperienceBuffer, make_env, plot_learning_curve
from DQNAgent import DQNAgent

SCORE_UPDATE = 100
N_GAMES = 10000
BATCH_SIZE = 10

env = make_env('Pong-v0')

if __name__ == "__main__":
    actions_n = env.action_space.n
    state_shape = env.observation_space.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DQNAgent(lr=0.0001, actions_n=actions_n, state_shape=state_shape)
    exp_buffer = ExperienceBuffer(1000)
    scores = []
    score_means = []
    eps_history = []

    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.choose_action(obs[np.newaxis, ...])
            next_obs, reward, done, _ = env.step(action)
            exp_buffer.insert(obs, action, reward, next_obs, done)

            if len(exp_buffer.states) % BATCH_SIZE == 0:
                for k in range(BATCH_SIZE):
                    exp = exp_buffer.sample()
                    agent.learn(exp[0][np.newaxis, ...],
                                exp[1], exp[2],
                                exp[3][np.newaxis, ...],
                                exp[4])
                agent.update_Q_tgt()

            score += reward
            obs = next_obs
        scores.append(score)
        eps_history.append(agent.epsilon)
        if i % SCORE_UPDATE == 0:
            mean_score = np.mean(scores[-SCORE_UPDATE:])
            score_means.append(mean_score)
            print("episode: {} \tmean score: {:.3} \tepsilon: {:.5}".format(i, mean_score, agent.epsilon))

    x = [i + 1 for i in range(N_GAMES)]
    plot_learning_curve(x, scores, eps_history)


