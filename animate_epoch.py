import matplotlib.pyplot as plt
from matplotlib import animation

from configs import DEFAULT_WORLD_DIM

colors = ['g', 'b', 'y', 'r'] * 4

dpi = 300
fig = plt.figure()
fig.set_dpi(dpi)
fig.set_size_inches(7, 6.5)
ax = plt.axes(xlim=(0, DEFAULT_WORLD_DIM), ylim=(0, DEFAULT_WORLD_DIM))
plt.tight_layout()
circles = []


def _update(t):
    locations = t['locations']
    global circles
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
    # only consider a particular batch
    batch = 99
    for timestep in timesteps:
        timestep['movements'] = timestep['movements'][batch]
        timestep['utterances'] = timestep['utterances'][batch]
        timestep['locations'] = timestep['locations'][batch]

    anim = animation.FuncAnimation(
        fig, _update, frames=timesteps, repeat=False)
    anim.save(output_filename, dpi=dpi)