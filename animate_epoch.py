import torch
from matplotlib import animation, patches
from matplotlib import pyplot as plt

from configs import DEFAULT_WORLD_DIM

base_colors = [
    'red',
    'blue',
    'green',
    'yellow',
    'orange',
    'pink',
    'grey',
]

COLORS = ['xkcd:{}'.format(color) for color in base_colors]
COLORS.extend(['xkcd:light {}'.format(color) for color in base_colors])
COLORS.extend(['xkcd:dark {}'.format(color) for color in base_colors])

artists = []

count = 0


def _update(t, num_agents, ax):
    sorted_goals = t['sorted_goals']
    goal_entities = t['goal_entities']
    locations = t['locations']
    physical = t['physical']

    def get_color(idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return COLORS[idx]

    global artists
    global count
    ax.set_title('timestep: {}'.format(count), fontsize=16)

    count += 1
    if not artists:
        for idx, loc in enumerate(locations):
            loc_list = loc.tolist()
            if idx < num_agents:
                goal = goal_entities[idx]
                patch = patches.Circle(
                    loc_list, radius=0.3, fc=get_color(goal))
            else:
                patch = patches.Rectangle(
                    loc_list, width=0.2, height=0.2, fc=get_color(idx))
            artists.append(ax.add_patch(patch))
    else:
        for idx, artist in enumerate(artists):
            loc_list = locations[idx].tolist()
            if isinstance(artist, patches.Circle):
                artist.set_center(loc_list)
            elif isinstance(artist, patches.Rectangle):
                artist.set_xy(loc_list)
            else:
                raise Exception("artist should be circle or rectangle")

    # movements = t['movements']
    # loss = t['loss']
    # utterances = t['utterances']
    # TODO: get agent goals


def animate(timesteps, output_filename, num_agents):
    fig = plt.figure(figsize=(20, 20))
    plt.subplots_adjust(left=0.2, right=0.8, top=0.75, bottom=0.25)
    fig.set_size_inches(20, 20)
    ax = plt.axes(xlim=(0, DEFAULT_WORLD_DIM), ylim=(0, DEFAULT_WORLD_DIM))

    global artists
    global count
    artists = []
    count = 0
    # only consider a particular batch
    batch = 99

    def pick_batch(timestep, key):
        timestep[key] = timestep[key][batch]

    timestep_keys = ('movements', 'utterances', 'locations', 'goal_entities',
                     'physical', 'sorted_goals')
    for timestep in timesteps:
        for key in timestep_keys:
            pick_batch(timestep, key)

    anim = animation.FuncAnimation(
        fig, _update, fargs=(num_agents, ax), frames=timesteps, repeat=False)
    writer = animation.writers['ffmpeg'](fps=2)
    anim.save(output_filename, writer=writer)


bee_artists = []
bee_count = 0


def _updateBee(t, num_agents, ax):
    locations = t['locations']
    physical = t['physical']
    votes = t['votes']
    hive_mask = t['hive_mask']
    hive_values = t['hive_values']
    # import ipdb
    # ipdb.set_trace()

    _, agent_vote = votes.max(1)
    agent_vote += num_agents

    def get_color(idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return COLORS[idx]

    global bee_artists
    global bee_count
    ax.set_title('timestep: {}'.format(bee_count), fontsize=16)

    bee_count += 1
    if not bee_artists:
        for idx, loc in enumerate(locations):
            loc_list = loc.tolist()
            if idx < num_agents:
                vote = agent_vote[idx]
                patch = patches.Circle(
                    loc_list, radius=0.3, fc=get_color(vote))
            else:
                hidden = hive_mask[idx - num_agents].item() == 0
                plt.text(
                    loc_list[0],
                    loc_list[1] - 0.5,
                    'value: {:.3f}{}'.format(
                        hive_values[idx - num_agents].item(),
                        ' (hidden)' if hidden else ''),
                    size=10,
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round",
                        ec=(1., 0.5, 0.5),
                        fc=(1., 0.8, 0.8),
                    ))
                patch = patches.Rectangle(
                    loc_list, width=0.2, height=0.2, fc=get_color(idx))
            bee_artists.append(ax.add_patch(patch))
    else:
        for idx, artist in enumerate(bee_artists):
            if idx < num_agents:
                vote = agent_vote[idx]
                artist.set_color(get_color(vote))
            loc_list = locations[idx].tolist()
            if isinstance(artist, patches.Circle):
                artist.set_center(loc_list)
            elif isinstance(artist, patches.Rectangle):
                artist.set_xy(loc_list)
            else:
                raise Exception("artist should be circle or rectangle")


def animateBee(timesteps, output_filename, num_agents):
    fig = plt.figure(figsize=(20, 20))
    plt.subplots_adjust(left=0.2, right=0.8, top=0.75, bottom=0.25)
    fig.set_size_inches(20, 20)
    ax = plt.axes(xlim=(0, DEFAULT_WORLD_DIM), ylim=(0, DEFAULT_WORLD_DIM))

    batch = 99

    def pick_batch(timestep, key):
        timestep[key] = timestep[key][batch]

    timestep_keys = ('movements', 'utterances', 'locations', 'votes',
                     'hive_mask', 'hive_values', 'physical')
    for timestep in timesteps:
        for key in timestep_keys:
            pick_batch(timestep, key)

    anim = animation.FuncAnimation(
        fig,
        _updateBee,
        fargs=(num_agents, ax),
        frames=timesteps,
        repeat=False)
    writer = animation.writers['ffmpeg'](fps=2)
    anim.save(output_filename, writer=writer)