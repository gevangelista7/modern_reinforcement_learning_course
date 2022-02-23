import torch
import numpy as np
from QFunctionNN import QFunctionNN
from utils import ExperienceBuffer, make_env, plot_learning_curve
from DQNAgent import DQNAgent

SCORE_UPDATE = 1
N_GAMES = 500

TRAINING = True

env = make_env('PongNoFrameskip-v4')

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
    game_steps = 0
    best_score = - np.inf
    print("Device: ", agent.device, "\n")
    for i in range(N_GAMES):
        obs = env.reset()
        game_score = 0
        done = False
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

        avg_score = np.mean(scores[-SCORE_UPDATE:])
        if avg_score > best_score:
            best_score = avg_score
            if TRAINING:
                agent.save_model()

        if i % SCORE_UPDATE == 0:
            print("episode: {} \tsteps: {} \tbest score: {:.3} \tscore: {:.3} \tepsilon: {:.5}"
                  .format(i, steps[-1], best_score, avg_score, agent.epsilon))

    plot_learning_curve(steps, scores, eps_history)


