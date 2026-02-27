import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from tqdm.auto import tqdm
import cv2
import re
import pickle
import numpy as np
import argparse


# for imagemagick:
from os import popen
from matplotlib import rcParams, tri

from multiprocessing import Pool
from functools import partial

rcParams["animation.convert_path"] = popen("which convert").readline().strip()
from warnings import warn


def plot_rollout(
    rollout_data,  # time x node x feature array
    triangulation,  # matplotlib.tri.Triangulation
    timepoints=None,  # list/array of length len(rollout_data)
    limits=None,  # 2 x n
    fps=20,
    writer="ffmpeg",  # ffmpeg, imagemagick, pillow
    output_file="temp.gif",
):
    """
    Plots/saves rollout

    rollout_data: time x node x feature array
    triangulation: matplotlib.tri.Triangulation object
    timepoints: list/array of length len(rollout_data)
    limits: 2 x feature array of min/max feature values to plot
    fps: frames per second
    writer: writer
    output_file: output file name
    """

    num_timepoints, num_cells, n = rollout_data.shape

    if timepoints is None:
        timepoints = range(num_timepoints)
    assert (len(timepoints)) == num_timepoints

    if limits is None:
        min_vals = np.min(rollout_data, axis=(0, 1))
        max_vals = np.max(rollout_data, axis=(0, 1))
    else:
        min_vals, max_vals = limits  # 2 x n
    assert len(min_vals) == n
    assert len(max_vals) == n

    if writer == "imagemagick" and rcParams["animation.convert_path"] == "":
        warn("imagemagick not available; switching to pillow")
        writer = "pillow"

    fig, ax = plt.subplots(n, 1, figsize=(10.7, 2 * n))
    plt.tight_layout()

    for i in range(n):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].axis("off")
        ax[i].triplot(triangulation, linewidth=0.1, color="black")

    def animate_func(t):
        for i in range(n):
            ax[i].tripcolor(
                triangulation,
                rollout_data[t, :, i],
                cmap="RdYlBu_r",
                vmin=min_vals[i],
                vmax=max_vals[i],
            )
        ax[0].set_title(timepoints[t])

        return ax

    anim = FuncAnimation(
        fig,
        animate_func,
        frames=len(rollout_data),
        blit=False,
        repeat=False,
        interval=200,
        cache_frame_data=False,
    )
    anim.save(output_file, writer=writer, fps=fps)

    return anim


def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

def create_video(image_folder, output_video, output_resolution=(1280, 1280), fps=50):
    #using existing .jpg or .png in this folder to create vedio
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort(key=numerical_sort)  # Sort images numerically

    width, height = output_resolution
 
    save_dir = f'{image_folder}/{output_video}'
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec for MP4
    video = cv2.VideoWriter(save_dir, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)

        # Resize frame to the desired output resolution
        resized_frame = cv2.resize(frame, (width, height))

        video.write(resized_frame)

    video.release()
    cv2.destroyAllWindows()

def plot_frame(
    triangulation, save_dir, state, classify, dpi, min_vals, max_vals, n, timepoints, t_and_data):
    t = t_and_data[0]
    data = t_and_data[1]
    # Subplots
    fig, ax = plt.subplots(n, 1, figsize=(6, 6 * n), dpi=dpi)
    plt.tight_layout()

    if n == 1:
        ax = [ax]
        state_x = state
        state_y = state
    else:
        state_x = '{}_x'.format(state)
        state_y = '{}_y'.format(state)

    for i in range(n):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].axis("on")
        ax[i].triplot(triangulation, linewidth=0.1, color="black")
        ax[i].tripcolor(
            triangulation,
            data[:, i],
            cmap="viridis",
            vmin=min_vals[i],
            vmax=max_vals[i],
            edgecolors='k',
            linewidth=0.1
        )
        if i == 0:
            ax[i].set_title('{} {} {}'.format(classify, state_x, timepoints[t]))
        if i == 1:
            ax[i].set_title('{} {} {}'.format(classify, state_y,  timepoints[t]))

    fig.savefig(f"{save_dir}/{t}_{n}_{classify}_{state}.jpg", bbox_inches="tight", dpi=dpi)
    plt.close()


def plot_rollout_separate(
    data,
    triangulation,
    state,
    classify,
    save_dir,
    fps=20,
    outname="rollout.mp4",
    dpi=600,
    timepoints=None,
    limits=None,
):
    """
    :param data: (time, node, dim)
    :param triangulation:
    # triangulation = matplotlib.tri.Triangulation(graph.pos[:, 0], graph.pos[:, 1]), where graph is a PyTorch Geometric object
    :param state: name of output state ( 'velocity' or 'pressure' )
    :param save_dir: where to save tmp plots as well as target movie
    :param fps: frame per second
    :param outname: name of output movie file, should probably end with '.mp4'
    :param dpi:
    :param timepoints: None
    :param limits: if not None, then
    # min_vals, max_vals = limits  # 2 x dim
    :return: None, target movie is saved to save_dir/outname
    """

    # data: (time, node, dim)
    # Timepoints
    num_timepoints, num_cells, n = data.shape

    if timepoints is None:
        timepoints = range(num_timepoints)
    assert (len(timepoints)) == num_timepoints

    # Limits
    if limits is None:
        min_vals = np.min(data, axis=(0, 1))
        max_vals = np.max(data, axis=(0, 1))
    else:
        min_vals, max_vals = limits  # 2 x n
    assert len(min_vals) == n
    assert len(max_vals) == n


    n_processes = int(len(os.sched_getaffinity(0)) * 0.7)
    pool = Pool(processes=n_processes)
    list(
        tqdm(
            pool.imap(
                partial(
                    plot_frame,
                    triangulation,
                    save_dir,
                    state,
                    classify,
                    dpi,
                    min_vals,
                    max_vals,
                    n,
                    timepoints,
                ),
                zip(timepoints, data),
            ),
            total=num_timepoints,
        )
    )

    create_video(save_dir, outname, output_resolution=(1280, 720), fps=fps)




#def plot_rollout_separate(
#    data,
#    triangulation,
#    save_dir,
#    fps=24,
#    outname="rollout.mp4",
#    dpi=200,
#    timepoints=None,
#    limits=None,
#):
#    """
#    :param data: (time, node, dim)
#    :param triangulation:
#    # triangulation = matplotlib.tri.Triangulation(graph.pos[:, 0], graph.pos[:, 1]), where graph is a PyTorch Geometric object
#    :param save_dir: where to save tmp plots as well as target movie
#    :param fps: frame per second
#    :param outname: name of output movie file, should probably end with '.mp4'
#    :param dpi:
#    :param timepoints: None
#    :param limits: if not None, then
#    # min_vals, max_vals = limits  # 2 x dim
#    :return: None, target movie is saved to save_dir/outname
#    """
#
#    def combine_into_movie(fps):
#        os.system(
#            f"ffmpeg -r {fps} -i {save_dir}/%01d_{n}.jpg -vcodec mpeg4 -y {save_dir}/{outname}"
#        )
#
#    # data: (time, node, dim)
#    # Timepoints
#    num_timepoints, num_cells, n = data.shape
#
#    if timepoints is None:
#        timepoints = range(num_timepoints)
#    assert (len(timepoints)) == num_timepoints
#
#    # Limits
#    if limits is None:
#        min_vals = np.min(data, axis=(0, 1))
#        max_vals = np.max(data, axis=(0, 1))
#    else:
#        min_vals, max_vals = limits  # 2 x n
#    assert len(min_vals) == n
#    assert len(max_vals) == n
#
#    n_processes = int(len(os.sched_getaffinity(0)) * 0.7)
#    pool = Pool(processes=n_processes)
#    list(
#        tqdm(
#            pool.imap(
#                partial(
#                    plot_frame,
#                    triangulation,
#                    save_dir,
#                    dpi,
#                    min_vals,
#                    max_vals,
#                    n,
#                    timepoints,
#                ),
#                zip(timepoints, data),
#            ),
#            total=num_timepoints,
#        )
#    )
#
#    combine_into_movie(fps=fps)

#    # Remove all tmp .jpg
#    os.system(f"rm {save_dir}/*_{n}.jpg")
#
#
#def plot_frame(
#    triangulation, save_dir, dpi, min_vals, max_vals, n, timepoints, t_and_data
#):
#    t = t_and_data[0]
#    data = t_and_data[1]
    # Subplots
#    fig, ax = plt.subplots(n, 1, figsize=(5, 2 * n), dpi=dpi)
#    plt.tight_layout()
#
#    if n == 1:
#        ax = [ax]
#
#    for i in range(n):
#        ax[i].set_xticks([])
#        ax[i].set_yticks([])
#        ax[i].axis("off")
#        ax[i].triplot(triangulation, linewidth=0.1, color="black")
#        ax[i].tripcolor(
#            triangulation,
#            data[:, i],
#            cmap="RdYlBu_r",
#            vmin=min_vals[i],
#            vmax=max_vals[i],
#        )
#
#    ax[0].set_title(timepoints[t])
#
#    fig.savefig(f"{save_dir}/{t}_{n}.jpg", bbox_inches="tight", dpi=dpi)
#    plt.close()


def interval_plot(pred, act, coords, timepoints):
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1])
    n = len(timepoints)
    min_val = np.min(np.concatenate((pred, act)))
    max_val = np.max(np.concatenate((pred, act)))
    # Subplots
    fig, axes = plt.subplots(n, 2, figsize=(10, 2 * n), dpi=200)
    plt.tight_layout()

    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
            ax.set_title(f'{"Predicted" if j==0 else "Actual"} t={timepoints[i]}')
            ax.triplot(triangulation, linewidth=0.1, color="black")
            ax.tripcolor(
                triangulation,
                pred[i] if j == 0 else act[i],
                cmap="RdYlBu_r",
                vmin=min_val,
                vmax=max_val,
            )
    return fig
