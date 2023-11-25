import math
import warnings

import gym
import numpy as np
import pandas as pd
from algorithms.planner import Planner
from algorithms.rl import RL
from examples.blackjack import Blackjack
from gym.envs.toy_text.frozen_lake import generate_random_map
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns



def main():
    main1()
    main2()
def main1():
    x = range(2, 21, 2)
    y=[]
    for i in x:
        # code modified from https://gymnasium.farama.org/environments/toy_text/frozen_lake/
        random_map = generate_random_map(size=i)
        frozen_lake = gym.make('FrozenLake-v1', render_mode=None, desc=random_map)
        # end
        # code from https://github.com/jlm429/bettermdptools/blob/master/examples/plots.py
        V, V_track, pi = Planner(frozen_lake.P).policy_iteration()
        # end
        y.append(V.sum()/len(V))
    plt.xlabel('iterations')
    plt.ylabel('Mean V')
    plt.plot(x, y)
    plt.legend()
    plt.show()
    # code modified from https://gymnasium.farama.org/environments/toy_text/frozen_lake/
    random_map = generate_random_map(size=20)
    frozen_lake = gym.make('FrozenLake-v1', render_mode=None, desc=random_map)
    # end
    # Q-learning
    x = range(1, 101, 10)
    y=[]
    for i in x:
        # code modified from https://github.com/jlm429/bettermdptools/blob/master/examples/plots.py
        V, V_track, pi = Planner(frozen_lake.P).policy_iteration(n_iters=i)
        # end
        y.append(V.sum()/len(V))
    plt.xlabel('iterations')
    plt.ylabel('Mean V')
    plt.plot(x, y)
    plt.legend()
    plt.show()

    # code modified from https://github.com/jlm429/bettermdptools/blob/master/examples/plots.py
    V, V_track, pi = Planner(frozen_lake.P).policy_iteration(n_iters=40)
    # end
    # find out reshape origin
    grid_world_policy_plot(V, 'policy')



    #vi
    x = range(2, 21, 2)
    y=[]
    for i in x:
        # code modified from https://gymnasium.farama.org/environments/toy_text/frozen_lake/
        random_map = generate_random_map(size=i)
        frozen_lake = gym.make('FrozenLake-v1', render_mode=None, desc=random_map)
        # end
        # code modified from https://github.com/jlm429/bettermdptools/blob/master/examples/plots.py
        V, V_track, pi = Planner(frozen_lake.P).value_iteration(n_iters=10000)
        # end
        y.append(V.sum()/len(V))
    plt.xlabel('iterations')
    plt.ylabel('Mean V')
    plt.plot(x, y)
    plt.legend()
    plt.show()
    # code modified from https://gymnasium.farama.org/environments/toy_text/frozen_lake/
    random_map = generate_random_map(size=20)
    frozen_lake = gym.make('FrozenLake-v1', render_mode=None, desc=random_map)
    # end
    # Q-learning
    x = range(1, 801, 10)
    y=[]
    for i in x:
        # code modified from https://github.com/jlm429/bettermdptools/blob/master/examples/plots.py
        V, V_track, pi = Planner(frozen_lake.P).value_iteration(n_iters=i)
        # end
        y.append(V.sum()/len(V))
    plt.xlabel('iterations')
    plt.ylabel('Mean V')
    plt.plot(x, y)
    plt.legend()
    plt.show()
    # code modified from https://github.com/jlm429/bettermdptools/blob/master/examples/plots.py
    V, V_track, pi = Planner(frozen_lake.P).value_iteration(n_iters=700)
    # end
    # find out reshape origin
    grid_world_policy_plot(V, 'policy')

    #Q learning
    x = range(2, 21, 2)
    y=[]
    for i in x:
        # code modified from https://gymnasium.farama.org/environments/toy_text/frozen_lake/
        random_map = generate_random_map(size=i)
        frozen_lake = gym.make('FrozenLake-v1', render_mode=None, desc=random_map)
        # End
        # code modified from https://github.com/jlm429/bettermdptools/blob/master/examples/plots.py
        Q, V, pi, Q_track, pi_track = RL(frozen_lake.env).q_learning(n_episodes=10000)
        # end
        y.append(V.sum()/len(V))
    plt.xlabel('iterations')
    plt.ylabel('Mean V')
    plt.plot(x, y)
    plt.legend()
    plt.show()
    # code modified from https://gymnasium.farama.org/environments/toy_text/frozen_lake/
    random_map = generate_random_map(size=20)
    frozen_lake = gym.make('FrozenLake-v1', render_mode=None, desc=random_map)
    # End
    # Q-learning
    x = range(2, 802, 10)
    y=[]
    for i in x:
        # code modified from https://github.com/jlm429/bettermdptools/blob/master/examples/plots.py
        Q, V, pi, Q_track, pi_track = RL(frozen_lake.env).q_learning(n_episodes=i)
        # end
        y.append(V.sum()/len(V))
    plt.xlabel('iterations')
    plt.ylabel('Mean V')
    plt.plot(x, y)
    plt.legend()
    plt.show()
    # code modified from https://github.com/jlm429/bettermdptools/blob/master/examples/plots.py
    Q, V, pi, Q_track, pi_track = RL(frozen_lake.env).q_learning(n_episodes=800)
    # end
    # find out reshape origin
    grid_world_policy_plot(V, 'policy')

def main2():
    # code from https://github.com/jlm429/bettermdptools/blob/master/examples/blackjack.py
    frozen_lake = Blackjack()
    # End
    # Q-learning
    x = range(1, 101, 10)
    y=[]
    for i in x:
        # code modified from https://github.com/jlm429/bettermdptools/blob/master/examples/plots.py
        V, V_track, pi = Planner(frozen_lake.P).policy_iteration(n_iters=i)
        # End
        y.append(V.sum()/len(V))
    plt.xlabel('iterations')
    plt.ylabel('Mean V')
    plt.plot(x, y)
    plt.legend()
    plt.show()
    # code modified from https://github.com/jlm429/bettermdptools/blob/master/examples/plots.py# cite planners
    V, V_track, pi = Planner(frozen_lake.P).policy_iteration(n_iters=20)
    # End
    # code modified from https://www.pythonpool.com/matplotlib-heatmap/
    plt.imshow(V.reshape(10,29))
    plt.colorbar()
    plt.show()
    # End


    # vi
    # code from https://github.com/jlm429/bettermdptools/blob/master/examples/blackjack.py
    frozen_lake = Blackjack()
    # End

    # Q-learning
    x = range(1, 401, 10)
    y=[]
    for i in x:
        # code modified from https://github.com/jlm429/bettermdptools/blob/master/examples/plots.py
        V, V_track, pi = Planner(frozen_lake.P).value_iteration(n_iters=i)
        # End
        y.append(V.sum()/len(V))
    plt.xlabel('iterations')
    plt.ylabel('Mean V')
    plt.plot(x, y)
    plt.legend()
    plt.show()
    # code modified from https://github.com/jlm429/bettermdptools/blob/master/examples/plots.py
    V, V_track, pi = Planner(frozen_lake.P).value_iteration(n_iters=400)
    # End
    # code modified from https://www.pythonpool.com/matplotlib-heatmap/
    plt.imshow(V.reshape(10,29))
    plt.colorbar()
    plt.show()
    # End
    # code from https://github.com/jlm429/bettermdptools/blob/master/examples/blackjack.py
    frozen_lake = Blackjack()
    # End

    #Q-learning
    x = range(2, 10001, 20)
    y=[]
    for i in x:
        # code modified from https://github.com/jlm429/bettermdptools/blob/master/examples/blackjack.py
        Q, V, pi, Q_track, pi_track = RL(frozen_lake.env).q_learning(frozen_lake.n_states, frozen_lake.n_actions, frozen_lake.convert_state_obs, n_episodes=i)
        # End
        y.append(V.sum()/len(V))
    plt.xlabel('iterations')
    plt.ylabel('Mean V')
    plt.plot(x, y)
    plt.legend()
    plt.show()
    # code modified from https://github.com/jlm429/bettermdptools/blob/master/examples/blackjack.py
    Q, V, pi, Q_track, pi_track = RL(frozen_lake.env).q_learning(frozen_lake.n_states, frozen_lake.n_actions, frozen_lake.convert_state_obs, n_episodes=10000)
    # End
    # code modified from https://www.pythonpool.com/matplotlib-heatmap/
    plt.imshow(V.reshape(10,29))
    plt.colorbar()
    plt.show()
    # End

# code from https://github.com/jlm429/bettermdptools/blob/master/examples/plots.py
def grid_world_policy_plot(data, label):
    if not math.modf(math.sqrt(len(data)))[0] == 0.0:
        warnings.warn("Grid map expected.  Check data length")
    else:
        data = np.around(np.array(data).reshape((20, 20)), 2)
        df = pd.DataFrame(data=data)
        my_colors = ((0.0, 0.0, 0.0, 1.0), (0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
        cmap = LinearSegmentedColormap.from_list('Custom', my_colors, len(my_colors))
        ax = sns.heatmap(df, cmap=cmap, linewidths=1.0)
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([.4, 1.1, 1.9, 2.6])
        colorbar.set_ticklabels(['Left', 'Down', 'Right', 'Up'])
        plt.title(label)
        plt.show()
# end

if __name__ == '__main__':
    main()

