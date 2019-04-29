import matplotlib.pyplot as plt
from matplotlib import animation

from configs import DEFAULT_WORLD_DIM

# only consider a particular batch
batch = 99

colors = ['g', 'b', 'y', 'r'] * 4

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)
ax = plt.axes(xlim=(0, DEFAULT_WORLD_DIM), ylim=(0, DEFAULT_WORLD_DIM))

circles = []


def _update(t):
    locations = t['locations'][batch]
    global circles
    if not circles:
        circles = [
            ax.add_patch(plt.Circle(loc.tolist(), radius=0.3, fc=colors[idx]))
            for idx, loc in enumerate(locations)
        ]
    else:
        for idx, circle in enumerate(circles):
            circle.center = locations[idx].tolist()
    # movements = t['movements'][batch]
    # loss = t['loss']
    # utterances = t['utterances'][batch]


def animate(timesteps):
    anim = animation.FuncAnimation(
        fig, _update, frames=timesteps, repeat=False, interval=1000)
    anim.save('epoch_animation.mp4')