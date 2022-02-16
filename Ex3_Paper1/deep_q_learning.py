import torch
import numpy as np
from QFunctionNN import QFunctionNN
from utils import ExperienceBuffer, make_env, plot_learning_curve
from DQNAgent import DQNAgent

SCORE_UPDATE = 100
N_GAMES = 10000
BATCH_SIZE = 100
LEARN_IDX = 100
TGT_Q_UPDATE = 1000

env = make_env('PongNoFrameskip-v0')

if __name__ == "__main__":
    actions_n = env.action_space.n
    state_shape = env.observation_space.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DQNAgent(lr=0.0001, actions_n=actions_n, state_shape=state_shape,
                     mem_size=50000, batch_size=32, q_update=1000)
    scores = []
    steps = []
    steps_means = []
    score_means = []
    eps_history = []

    for i in range(N_GAMES):
        obs = env.reset()
        game_score = 0
        game_steps = 0
        done = False
        print("Device: ", agent.device, "\n")
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = env.step(action)

            agent.insert_memory(obs, action, reward, next_obs, done)
            agent.learn()

            game_score += reward
            game_steps += 1
            obs = next_obs

        scores.append(game_score)
        steps.append(game_steps)
        eps_history.append(agent.epsilon)
        if i % SCORE_UPDATE == 0:
            mean_score = np.mean(scores[-SCORE_UPDATE:])
            mean_steps = np.mean(steps[-SCORE_UPDATE:])
            score_means.append(mean_score)
            steps_means.append(mean_steps)
            print("episode: {} \tmean score: {:.3} \tmean steps: {:.3} \tepsilon: {:.5}"
                  .format(i, mean_score, mean_steps, agent.epsilon))

    x = [i + 1 for i in range(N_GAMES)]
    plot_learning_curve(x, scores, eps_history)


