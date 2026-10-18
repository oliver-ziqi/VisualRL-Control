import os
import subprocess
import pickle
import numpy as np
import matplotlib.pyplot as plt

def run_exp(name, args_list, env_overrides, game='Pendulum'):
    env = os.environ.copy()
    env.update(env_overrides)
    cmd = ['python', 'train.py', '--game', game] + args_list
    subprocess.run(cmd, env=env, check=True)
    base = os.path.dirname(os.path.realpath(__file__))
    res_path = os.path.join(base, 'data/plots/dqn_result_o_'+game+'.pkl')
    with open(res_path, 'rb') as f:
        y = pickle.load(f)
    return np.array(y)

def main():
    game = 'Pendulum'
    base = os.path.dirname(os.path.realpath(__file__))
    save_freq = 1000
    curves = []
    labels = []

    yA = run_exp('A', ['--learn_start','5000','--learn_freq','4','--eps_init','1.0','--eps_phase1_steps','50000','--eps_phase2_steps','100000','--save_freq',str(save_freq),'--n_envs','1'], {'DQN_LR':'1e-4','DQN_SOFT_UPDATE_RATE':'0.01'})
    curves.append(yA); labels.append('A')

    yB = run_exp('B', ['--learn_start','1000','--learn_freq','2','--eps_init','1.0','--eps_phase1_steps','30000','--eps_phase2_steps','70000','--save_freq',str(save_freq),'--n_envs','2'], {'DQN_LR':'3e-4','DQN_SOFT_UPDATE_RATE':'0.02'})
    curves.append(yB); labels.append('B')

    yC = run_exp('C', ['--learn_start','3000','--learn_freq','4','--eps_init','1.0','--eps_phase1_steps','40000','--eps_phase2_steps','80000','--save_freq',str(save_freq),'--n_envs','1'], {'DQN_BATCH':'128','DQN_LR':'1e-4','DQN_SOFT_UPDATE_RATE':'0.01','DQN_N_TILE':'15'})
    curves.append(yC); labels.append('C')

    plt.figure(figsize=(9,5))
    for y, lab in zip(curves, labels):
        x = np.arange(1, len(y)+1) * save_freq
        plt.plot(x, y, label=lab)
    plt.xlabel('Used steps')
    plt.ylabel('Mean ep 100 return')
    plt.title('DQN Experiments: '+game)
    plt.grid(True)
    plt.legend()
    out_path = os.path.join(base, 'data/plots/dqn_compare_'+game+'.png')
    plt.tight_layout()
    plt.savefig(out_path)

if __name__ == '__main__':
    main()
