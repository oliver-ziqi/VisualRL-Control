import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='Pendulum')
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--show', action='store_true', default=False)
    args = parser.parse_args()

    current_path = os.path.dirname(os.path.realpath(__file__))
    result_path = os.path.join(current_path, 'data/plots/dqn_result_o_'+args.game+'.pkl')
    plot_path = os.path.join(current_path, 'data/plots/dqn_curve_o_'+args.game+'_viz.png')

    if not os.path.isfile(result_path):
        raise FileNotFoundError('Result file not found: '+result_path)

    with open(result_path, 'rb') as f:
        result = pickle.load(f)

    y = np.array(result)
    x = np.arange(1, len(y)+1) * args.save_freq

    plt.figure(figsize=(8,4))
    plt.plot(x, y, label='Mean ep 100 return')
    plt.xlabel('Used steps')
    plt.ylabel('Mean ep 100 return')
    plt.title('Training Curve: '+args.game)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    if args.show:
        plt.show()

if __name__ == '__main__':
    main()
