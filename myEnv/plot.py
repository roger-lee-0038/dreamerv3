import pickle
import matplotlib.pyplot as plt
import rl_utils
import sys

f_name = sys.argv[1]
with open(f_name, "rb") as handler:
    env_name, return_list, agent = pickle.load(handler)

return_list = [sum(i) for i in return_list]
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()
