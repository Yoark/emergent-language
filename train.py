import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

import configs
from animate_epoch import animate, animateBee
from modules.agent import AgentModule
from modules.game import GameModule
from modules.beegame import BeeGameModule
from modules.bee import BeeModule

parser = argparse.ArgumentParser(
    description="Trains the agents for cooperative communication task")
parser.add_argument(
    '--no-utterances',
    action='store_true',
    help='if specified disables the communications channel (default enabled)')
parser.add_argument(
    '--penalize-words',
    action='store_true',
    help='if specified penalizes uncommon word usage (default disabled)')
parser.add_argument(
    '--bee-game',
    action='store_true',
    help='if specified penalizes uncommon word usage (default disabled)')
parser.add_argument(
    '--n-epochs',
    '-e',
    type=int,
    help='if specified sets number of training epochs (default 5000)')
parser.add_argument(
    '--learning-rate',
    type=float,
    default=1e-3,
    help='if specified sets learning rate (default 1e-3)')
parser.add_argument(
    '--batch-size',
    type=int,
    default=256,
    help='if specified sets batch size(default 256)')
parser.add_argument(
    '--n-timesteps',
    '-t',
    type=int,
    default=32,
    help='if specified sets timestep length of each episode (default 32)')
parser.add_argument(
    '--num-shapes',
    '-s',
    type=int,
    default=3,
    help='if specified sets number of colors (default 3)')
parser.add_argument(
    '--num-colors',
    '-c',
    type=int,
    default=3,
    help='if specified sets number of shapes (default 3)')
parser.add_argument(
    '--max-agents',
    type=int,
    default=3,
    help=
    'if specified sets maximum number of agents in each episode (default 3)')
parser.add_argument(
    '--min-agents',
    type=int,
    default=1,
    help=
    'if specified sets minimum number of agents in each episode (default 1)')
parser.add_argument(
    '--max-landmarks',
    type=int,
    default=3,
    help=
    'if specified sets maximum number of landmarks in each episode (default 3)'
)
parser.add_argument(
    '--num-swarm', type=int, default=10, help='set num of swarm')
parser.add_argument(
    '--num-scouts', type=int, default=2, help='set num of scouts')
parser.add_argument(
    '--num-hives', type=int, default=4, help='set num of hives')

parser.add_argument(
    '--min-landmarks',
    type=int,
    default=1,
    help=
    'if specified sets minimum number of landmarks in each episode (default 1)'
)
parser.add_argument(
    '--vocab-size',
    '-v',
    type=int,
    default=20,
    help='if specified sets maximum vocab size in each episode (default 20)')
parser.add_argument(
    '--world-dim',
    '-w',
    type=int,
    default=16,
    help=
    'if specified sets the side length of the square grid where all agents and landmarks spawn(default 16)'
)
parser.add_argument(
    '--oov-prob',
    '-o',
    type=int,
    default=1,
    help=
    'higher value penalize uncommon words less when penalizing words (default 1)'
)
parser.add_argument(
    '--load-model-weights',
    type=str,
    help=
    'if specified start with saved model weights saved at file given by this argument'
)
parser.add_argument(
    '--save-model-weights',
    type=str,
    help='if specified save the model weights at file given by this argument')
parser.add_argument(
    '--use-cuda',
    action='store_true',
    help='if specified enables training on CUDA (default disabled)')


def print_losses(epoch, losses, dists, game_config):
    for a in range(game_config.min_agents, game_config.max_agents + 1):
        for l in range(game_config.min_landmarks,
                       game_config.max_landmarks + 1):
            loss = losses[a][l][-1] if len(losses[a][l]) > 0 else 0
            min_loss = min(losses[a][l]) if len(losses[a][l]) > 0 else 0

            if dists:
                dist = dists[a][l][-1] if len(dists[a][l]) > 0 else 0
                min_dist = min(dists[a][l]) if len(dists[a][l]) > 0 else 0
            else:
                dist = -1
                min_dist = -1

            print(
                "[epoch %d][%d agents, %d landmarks][%d batches][last loss: %f][min loss: %f][last dist: %f][min dist: %f]"
                % (epoch, a, l, len(losses[a][l]), loss, min_loss, dist,
                   min_dist))
    print("_________________________")


def print_bee_losses(epoch, losses, game_config):
    swarm = game_config.num_swarm
    scout = game_config.num_scouts
    hive = game_config.num_hives

    loss = losses[swarm][scout][hive][-1] if len(
        losses[swarm][scout][hive]) > 0 else 0
    min_loss = min(losses[swarm][scout][hive]) if len(
        losses[swarm][scout][hive]) > 0 else 0

    print(
        "[epoch %d][%d swarm, %d scouts %d hives ][%d batches][last loss: %f][min loss: %f]"
        % (epoch, swarm, scout, hive, len(losses[swarm][scout][hive]), loss,
           min_loss))
    print("_________________________")


def main():
    args = vars(parser.parse_args())

    training_config = configs.get_training_config(args)
    print("Training with config:")
    print(training_config)

    if not args['bee_game']:
        losses = defaultdict(lambda: defaultdict(list))
        dists = defaultdict(lambda: defaultdict(list))
        game_config = configs.get_game_config(args)
        agent_config = configs.get_agent_config(args)
        print(game_config)
        print(agent_config)
        agent = AgentModule(agent_config)
        if training_config.use_cuda:
            agent = agent.cuda()
        optimizer = RMSprop(
            agent.parameters(), lr=training_config.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, 'min', verbose=True, cooldown=5)

        num_utters = []
        utter_times = []
        for epoch in range(training_config.num_epochs):
            num_agents = np.random.randint(game_config.min_agents,
                                           game_config.max_agents + 1)
            num_landmarks = np.random.randint(game_config.min_landmarks,
                                              game_config.max_landmarks + 1)
            agent.reset()
            game = GameModule(game_config, num_agents, num_landmarks)

            if training_config.use_cuda:
                game = game.cuda()
            optimizer.zero_grad()

            total_loss, timesteps, num_utter, utter_num_t, prob = agent(game)

            output_filename = 'epoch_{}_animation.mp4'.format(epoch)

            num_utters.append(num_utter)
            utter_times.append(torch.mean(torch.Tensor(utter_num_t)))

            if epoch % 10 == 0:
                print(prob)
                animate(timesteps, output_filename, num_agents)

            per_agent_loss = total_loss.item(
            ) / num_agents / game_config.batch_size
            losses[num_agents][num_landmarks].append(per_agent_loss)
            print(losses)

            dist = game.get_avg_agent_to_goal_distance()
            avg_dist = dist.item() / num_agents / game_config.batch_size
            dists[num_agents][num_landmarks].append(avg_dist)

            print_losses(epoch, losses, dists, game_config)

            total_loss.backward()
            optimizer.step()

            if num_agents == game_config.max_agents and num_landmarks == game_config.max_landmarks:
                scheduler.step(losses[game_config.max_agents][
                    game_config.max_landmarks][-1])
    else:

        game_config = configs.get_beegame_config(args)
        agent_config = configs.get_bee_config(args)
        print(game_config)
        print(agent_config)
        agent = BeeModule(agent_config)
        losses = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        if training_config.use_cuda:
            agent = agent.cuda()

        optimizer = RMSprop(
            agent.parameters(), lr=training_config.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, 'min', verbose=True, cooldown=5)

        num_utters = []
        utter_times = []
        num_swarm = game_config.num_swarm
        num_scouts = game_config.num_scouts
        num_hives = game_config.num_hives
        num_agents = num_swarm + num_scouts
        votes_ratio_per_ts = []
        votes_ratio_per_epoch = []
        for epoch in range(training_config.num_epochs):
            agent.reset()
            game = BeeGameModule(game_config, num_swarm, num_scouts, num_hives)
            if training_config.use_cuda:
                game = game.cuda()

            optimizer.zero_grad()

            total_loss, timesteps, num_utter, utter_num_t, prob, votes_epoch, votes_ratios = agent(
                game)

            output_filename = 'bee_game_epoch_{}_animation.mp4'.format(epoch)
            animateBee(timesteps, output_filename, num_agents)


            ratio = game.max_freq(votes_epoch).mean()

            votes_ratio_per_epoch.append(ratio)
            votes_ratio_per_ts.append(votes_ratios)

            num_utters.append(num_utter)
            utter_times.append(torch.mean(torch.Tensor(utter_num_t)))

            if epoch % 10 == 0:
                print(prob)

            per_agent_loss = total_loss.item(
            ) / num_agents / game_config.batch_size

            losses[num_swarm][num_scouts][num_hives].append(per_agent_loss)

            print_bee_losses(epoch, losses, game_config)

            total_loss.backward()
            optimizer.step()

            scheduler.step(losses[num_swarm][num_scouts][num_hives][-1])

    if training_config.save_model:
        #print(agent.LOG)
        torch.save(agent, training_config.save_model_file)
        print("Saved agent model weights at %s" %
              training_config.save_model_file)
    if args['bee_game']:
        return num_utters, utter_times, votes_ratio_per_epoch, votes_ratio_per_ts
    else:
        return num_utters, utter_times


if __name__ == "__main__":
    num_utters, utter_num_t, votes_ratio_per_epoch, votes_ratio_per_ts = main()
