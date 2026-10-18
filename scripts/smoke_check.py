"""Check network, update, environment and historical data consistency."""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    torch.set_num_threads(1)
    import dqn
    from wrappers import wrap_cover_pendulun
    dqn.USE_GPU=False
    retained_weights=torch.load(str(ROOT/'data/model/dqn_pred_net_o_Pendulum.pkl'),map_location='cpu',weights_only=True)
    dqn.N_TILE=(retained_weights['fc_0.weight'].shape[1]-6*6*8)//dqn.N_STATE
    assert dqn.N_TILE==15
    agent=dqn.DQN()
    state=[np.zeros((4,42,42),dtype=np.uint8),np.zeros(4,dtype=np.float32)]
    for i in range(dqn.BATCH_SIZE):
        agent.store_transition(state,i%9,-1.0,state,i%2)
    assert agent.choose_action([state],0,False).shape==(1,)
    loss=agent.learn()
    assert np.isfinite(loss)
    agent.load_model(str(ROOT/'data/model/dqn_pred_net_o_Pendulum.pkl'),str(ROOT/'data/model/dqn_target_net_o_Pendulum.pkl'))
    assert agent.choose_action([state],0,False).shape==(1,)
    actions=[np.array(x,dtype=np.float32) for x in np.linspace(-2,2,9)]
    env=wrap_cover_pendulun('Pendulum',actions,episode_length=4)()
    try:
        observation,_=env.reset()
        assert observation[0].shape==(4,42,42)
        assert observation[1].shape==(4,)
        finished=False
        for i in range(10):
            transition=env.step(4)
            if transition[2]:
                finished=True
                break
        assert finished
    finally:
        env.close()
    import pandas as pd
    log=pd.read_csv(ROOT/'artifacts/monitor.csv',comment='#')
    assert abs(log['r'].tail(100).mean()-(-198.41396088))<1e-6
    print('DQN forward/update, rendered observations, episode boundary and retained metric passed')

if __name__=='__main__':
    main()
