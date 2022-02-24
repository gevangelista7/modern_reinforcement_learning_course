import gym
from MCAgent import MCAgent

N_GAMES = 500000

if __name__ == "__main__":
    env = gym.make("Blackjack-v0")
    agent = MCAgent(env.observation_space, env.action_space)
    results = []
    for i in range(N_GAMES):
        obs = env.reset()
        done = False
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            agent.insert_memory(state=obs, action=action, reward=reward)
            agent.update_Q()
            obs = obs_
            results.append(reward)
        agent.update_policy()
        agent.clear_memory()
        if i % 1000 == 0:
            win_rate = results.count(1)/len(results)
            print("current ep: {}\t pi(21,2,True): {}\t win rate: {}"
                  .format(i, agent.policy[(21, 2, 1)], win_rate)) #sum(list(agent.policy.values()))))

    print("Final pi(21,2,True): {}".format(agent.policy[(21, 2, 1)]))