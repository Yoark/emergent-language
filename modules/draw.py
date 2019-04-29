import matplotlib.pyplot as plt
from matplotlib import animation
import torch

"""use it in colab"""
fig = plt.figure()
ax = plt.axes(xlim=(0, 21), ylim=(0,1))
plt.xticks(torch.arange(0,21))
barcollection = plt.bar(torch.arange(1, 21), probs[2][0].cpu().numpy())

def animate(i):
    x = torch.arange(1, 21)
    h = probs[2][i+1].cpu().numpy()
    for j, b in enumerate(barcollection):
        b.set_height(h[j])

anim = animation.FuncAnimation(fig, animate, repeat=False, frames=500, interval=100)
anim.save('bar_anime.mp4', writer=animation.FFMpegWriter(fps=10))
"""download it """
from google.colab import files
files.download('bar_anime.mp4')