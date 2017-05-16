#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

    python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --render --num_rollouts 1


Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import torch
import net as mynet 
import net_utils as nu
import os
#import train_pytorch as tp
import ipdb
epsilon = 0.00001
from torch.autograd import Variable
from IPython import embed
import numpy as np
observe_model = True
collect_dagger_data = True
collect_train_data = False

def load_model(envname, save_model_dir):
    env = gym.make(envname)
    out_size = env.action_space.shape[0]
    in_size = env.observation_space.shape[0]    
    model = mynet.Net(in_size, out_size, [4000, 500])
    model, ckpt = nu.load_and_test(model, save_model_dir)
    return model, ckpt

def load_mean_std(save_model_dir):
    mean_std_file = save_model_dir + '/' + 'mean_std.pk'
    with open(mean_std_file,'rb') as fp2:
        mean_std = pickle.load(fp2)
    return mean_std['mean'], mean_std['std']
    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=2,
                        help='Number of expert roll outs')
    args = parser.parse_args()


def run_eval(model=None, args):
    if model is None:
        save_model_dir = './trained_model/' + args.envname
        model,ckpt = load_model(args.envname, save_model_dir)
        mean, std = load_mean_std(save_model_dir)
        model.eval()
    
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        # env.spec.timestep_limit = args.max_timesteps
        # env.spec.trials = 400
        max_steps = args.max_timesteps or env.spec.timestep_limit
        # ipdb.set_trace()
        returns = []
        observations = []
        actions = []
        gt_actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                gt_action = policy_fn(obs[None,:])
                if observe_model:
                    x_batch = (obs - mean)/(std + epsilon)
                    x_batch = np.expand_dims(x_batch,axis=0)
                    x_batch = np.asarray(x_batch,dtype=np.float32)
                    input = Variable(torch.from_numpy(x_batch))
                    output = model(input)
                    action = output.data.numpy()
                else:
                    action = gt_action
                    
                observations.append(obs)
                actions.append(action)
                gt_actions.append(gt_action)
                
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if 0 and steps >= max_steps:
                    break
            returns.append(totalr)

        print('steps:{} max_steps:{}'.format(steps, max_steps))
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        
        expert_data = {'observations': np.array(observations),
                       'actions': np.array(gt_actions),
                       'model_actions':np.array(actions)}
        if 0:
            pk_name = args.envname + '_roll{}'.format(args.num_rollouts) + '.pk'
            with open(pk_name, "wb") as output_file:
                pickle.dump(expert_data, output_file)
                
        if collect_dagger_data:
            pk_name = args.envname + 'dagger_roll{}'.format(args.num_rollouts) + '.pk'
            with open(pk_name, "wb") as output_file:
                pickle.dump(expert_data, output_file)
            
        return expert_data    

if __name__ == '__main__':
    main()
