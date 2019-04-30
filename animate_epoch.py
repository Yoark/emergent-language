import torch
from matplotlib import animation, patches
from matplotlib import pyplot as plt

from configs import DEFAULT_WORLD_DIM

fig = plt.figure(figsize=(20, 20))
plt.subplots_adjust(left=0.2, right=0.8, top=0.75, bottom=0.25)
fig.set_size_inches(20, 20)
ax = plt.axes(xlim=(0, DEFAULT_WORLD_DIM), ylim=(0, DEFAULT_WORLD_DIM))
circles = []

count = 0


def _update(t, num_agents):
    sorted_goals = t['sorted_goals']
    goal_entities = t['goal_entities']
    locations = t['locations']
    physical = t['physical']

    def get_color(idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return 'C{}'.format(idx)

    global circles
    global count
    ax.set_title('timestep: {}'.format(count), fontsize=16)

    count += 1
    if not circles:
        for idx, loc in enumerate(locations):
            loc_list = loc.tolist()
            if idx < num_agents:
                goal = goal_entities[idx]
                patch = patches.Circle(
                    loc_list, radius=0.3, fc=get_color(goal))
            else:
                patch = patches.Rectangle(
                    loc_list, width=0.2, height=0.2, fc=get_color(idx))
            circles.append(ax.add_patch(patch))
    else:
        for idx, circle in enumerate(circles):
            circle.center = locations[idx].tolist()
    # movements = t['movements']
    # loss = t['loss']
    # utterances = t['utterances']
    # TODO: get agent goals


def animate(timesteps, output_filename, num_agents):
    global circles
    global count
    circles = []
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
        fig, _update, fargs=(num_agents, ), frames=timesteps, repeat=False)
    writer = animation.writers['ffmpeg'](fps=2)
    anim.save(output_filename, writer=writer)
