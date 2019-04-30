import matplotlib.pyplot as plt
from matplotlib import animation, rcParams

from configs import DEFAULT_WORLD_DIM

colors = ['g', 'b', 'y', 'r'] * 4

fig = plt.figure(figsize=(20, 20))
plt.subplots_adjust(left=0.2, right=0.8, top=0.75, bottom=0.25)
fig.set_size_inches(20, 20)
ax = plt.axes(xlim=(0, DEFAULT_WORLD_DIM), ylim=(0, DEFAULT_WORLD_DIM))
circles = []

count = 0


def _update(t):
    locations = t['locations']
    global circles, count
    ax.set_title('timestep: {}'.format(count), fontsize=16)
    count += 1
    if not circles:
        circles = [
            ax.add_patch(plt.Circle(loc.tolist(), radius=0.3, fc=colors[idx]))
            for idx, loc in enumerate(locations)
        ]
    else:
        for idx, circle in enumerate(circles):
            circle.center = locations[idx].tolist()
    # movements = t['movements']
    # loss = t['loss']
    # utterances = t['utterances']
    # TODO: get agent goals


def animate(timesteps, output_filename):
    global circles, count
    circles = []
    count = 0
    # only consider a particular batch
    batch = 99
    for timestep in timesteps:
        timestep['movements'] = timestep['movements'][batch]
        timestep['utterances'] = timestep['utterances'][batch]
        timestep['locations'] = timestep['locations'][batch]

    anim = animation.FuncAnimation(
        fig, _update, frames=timesteps, repeat=False)
    writer = animation.writers['ffmpeg'](fps=2)
    anim.save(output_filename, writer=writer)