import matplotlib.pylab as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import pandas as pd
import numpy as np

def Plot_3D(xyz, **kwargs):

    """
    Plot a rotateable 3d plot

    parameters
    ----------
    xyz: a data frame which has three columns as Z0, Z1, Z2 for plotting
    color: name of a column of data frame which has the colors
    annotation: True or False, if True the index values of data frame will be used
    theta1 : first rotation angle
    theta2 : second rotation angle
    ax_labels : list of axis labels
    plot_size: tuple of plot sizes
    """
    theta1 = kwargs.get('theta1', None)
    theta2 = kwargs.get('theta2', None)
    plot_size = kwargs.get('plot_size', (20, 20))
    plot_title = kwargs.get('plot_title', None)
    annotation_col = kwargs.get('annotation_col', None)

    fig = plt.figure(figsize=plot_size)
    ax = fig.add_subplot(111, projection='3d')
    x = xyz["Z0"]
    y = xyz["Z1"]
    z = xyz["Z2"]

    if 'cluster_color' not in xyz:
        c = "Blue"
    else:
        c = xyz["cluster_color"]

    ax.scatter(x, y, z, color=c, s=100)
    if annotation_col is not None:
        for i, txt in enumerate(xyz[annotation_col].tolist()):
            ax.text(x[i], y[i], z[i], txt, size=10)

    ax_labels = kwargs.get('ax_labels', None)
    if ax_labels is not None:
        ax.set_xlabel(ax_labels[0], fontsize=20)
        ax.set_ylabel(ax_labels[1], fontsize=20)
        ax.set_zlabel(ax_labels[2], fontsize=20)

    if theta1 and theta2 is not None:
        ax.view_init(theta1, theta2)

    if plot_title is not None:
        ax.set_title(plot_title)

    return ax


def MScatter_plot(datasets, plot_dim=2, **kwargs):

    """
    Takes multiple datasets and scatter plot each of them

    Parameters
    ----------
    datasets: a dict of dataframes, each of the dataframes going to be a scatter plot
    plot_dim: number of dimensions to plot (2 or 3)

    Optional parameters
    --------------------
    fig_size:  a tuple of fig size
    dataset_color_colname: the column to read the color of each point
    hspace : horizontal space between subplots
    wspace : width space between subplots
    scatter_point_size: scatter point size
    """

    fig_size = kwargs.get('fig_size', (10, len(datasets)*10))
    dataset_color_colname = kwargs.get('dataset_color_colname', 'cluster_color')
    hspace = kwargs.get('hspace', 0.5)
    wspace = kwargs.get('wspace', 0.5)
    scatter_point_size = kwargs.get('scatter_point_size', 10)


    fig = plt.figure(figsize=fig_size)
    i = 1
    for k, v in datasets.items():
        if plot_dim == 3:
            ax = fig.add_subplot(15, 3, i, projection="3d")
            if dataset_color_colname:
                ax.scatter(v['Z0'], v['Z1'], v['Z2'], c=v[dataset_color_colname], s=scatter_point_size)

        else:
            ax = fig.add_subplot(15, 3, i)
            if dataset_color_colname:
                ax.scatter(v['Z0'], v['Z1'], c=v[dataset_color_colname], s=scatter_point_size)

        ax.set_title(k)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        i += 1

    return fig


def draw_heatmap(*args, **kwargs):
    """
    Takes data frame and then make it an square matrix and plot a heatmap

    Parameters
    ----------
    data: a data frame with at least 3 cols
    index: this is going to be the x axis on the heatmap
    column: this is going to be the y axis on the heatmap
    value: this is the weight or color of heatmap

    return
    --------------------
    a heatmap
    """
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs)

def Facet_Grid_Heatmap(data, groupby_col, col_wrap, height, index, column, value):
    """
    Takes a data frame, group it based on a column and plot a heatmap for each group

    Parameters
    ----------
    data: a data frame with at least 3 cols
    index: this is going to be the x axis on the heatmap
    column: this is going to be the y axis on the heatmap
    value: this is the weight or color of heatmap
    groupby_col: is the groupby column
    col_wrap: number of heat map in each row when plotting
    height: height of each heatmap

    return
    --------------------
    a heatmap grid
    """

    fg = sns.FacetGrid(data, col=groupby_col, col_wrap = col_wrap, height= height)
    fg.map_dataframe(draw_heatmap, index, column, value, cbar=False, square = True)

    for ax in fg.axes.flat:
        ax.set_aspect('equal','box')
    plt.show()

def plot_loss(loss_filepath):
    """

    Parameters
    ----------
    loss_filename
    loss_filedir

    Returns
    -------

    """
    data = pd.read_csv(loss_filepath, header=None)
    data.columns = ["epochs", "loss"]

    plt.figure(figsize=(10, 5))
    plt.plot(data["epochs"], data["loss"])
    plt.xlabel("epochs")
    plt.ylabel("loss")

    return plt

def plot_embedding(data, plot_dim, **kwargs):
    """

    Parameters
    ----------
    data: a dataframe which containes the Z0, Z1, Z2 ... and colors and annotations
    plot_dim: 2d or 3d plot
    scatter_point_size(optional): size of the scatter points
    annotation: if True, it will use the indexes for annotation

    Returns
    -------

    """
    plot_size = kwargs.get('plot_size', (10, 10))
    annotation = kwargs.get('annotation', False)
    theta1 = kwargs.get('theta1', None)
    theta2 = kwargs.get('theta2', None)
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    zlim = kwargs.get('zlim', None)
    alpha = kwargs.get('alpha', None)
    scatter_point_size = kwargs.get('scatter_point_size', 40)
    data.index = data.index.astype('str')



    fig = plt.figure(figsize=plot_size)

    if plot_dim == 2:
        ax = fig.add_subplot(111)
        ax.scatter(data['Z0'], data['Z1'], color=data['cluster_color'], s=scatter_point_size, alpha=alpha)
        if annotation:
            for j, txt in enumerate(data.index.tolist()):
                ax.text(data['Z0'][j], data["Z1"][j], txt, size=10)
    if plot_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data['Z0'], data['Z1'], data["Z2"], color=data['cluster_color'], s=scatter_point_size, alpha=alpha)
        if annotation:
            for j, txt in enumerate(data.index.tolist()):
                ax.text(data['Z0'][j], data["Z1"][j], data["Z2"][j], txt, size=10)
        if theta1 and theta2 is not None:
            ax.view_init(theta1, theta2)

    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if zlim:
        ax.set_zlim(zlim[0], zlim[1])


    for tick in ax.xaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(12)

    for tick in ax.yaxis.get_majorticklabels():  # example for xaxis
        tick.set_fontsize(12)



    return fig


def plot_polar_source_target_relation(theta, r1, r2=None, **kwargs):
    """

    parameters
    ----------
    title: string
    title_position: list of two numbers for x and y position of title
    theta: the x values
    r1: np.array y1 valus
    r2: np.array y2 valus
    r1_line_color: color of the r1 plot lines
    r1_scatter_colors: colors of the r1 plot scatter points
    r2_line_color: color of the r2 plot lines
    r2_scatter_colors: colors of the r2 plot scatter points
    xtick_labels: labels of the xticks
    xtick_labels_font: fonts of of the xticks labels
    right_labels: list of index of all right labels
    left_labels: list of index of all left labels
    """

    n_nodes = len(theta)

    plot_size = kwargs.get('plot_size', (10, 10))
    title = kwargs.get('title', None)
    title_position = kwargs.get('title_position', [.5, 1.25])
    r1_line_color = kwargs.get('r1_line_color', 'cornflowerblue')
    r1_scatter_colors = kwargs.get('r1_scatter_colors', ['cornflowerblue'])
    r2_line_color = kwargs.get('r2_line_color', 'red')
    r2_scatter_colors = kwargs.get('r2_scatter_colors', ['red'])
    xtick_labels = kwargs.get('xtick_labels', [str(i) for i in range(0, n_nodes)])
    xtick_labels_font = kwargs.get('xtick_labels_font', 10)
    xtick_label_colors = kwargs.get('xtick_label_colors', ['black'])
    right_labels = kwargs.get('right_labels', [i for i in range(24, 70)])
    left_labels = kwargs.get('left_labels', [i for i in range(0, 24)] + [i for i in range(70, n_nodes)])
    print_yticklabels = kwargs.get('print_yticklabels', False)
    rmax = kwargs.get('rmax', 0.0006)


    fig = plt.figure(figsize=plot_size)
    ax1 = fig.add_subplot(121, projection='polar')

    if title:
        ax1.title.set_text("Emitter_" + title)
    if title_position:
        ax1.title.set_position(title_position)

    ax1.plot(theta, r1, c=r1_line_color, label="Emit")
    ax1.scatter(theta, r1, c=r1_scatter_colors)

    if r2.tolist():
        ax2 = fig.add_subplot(122, projection='polar')
        ax2.plot(theta, r2, c=r2_line_color, alpha=0.5, label="Receive")
        ax2.scatter(theta, r2, c=r2_scatter_colors, alpha=0.5)

    if title:
        ax2.title.set_text("Receiver_" + title)
    if title_position:
        ax2.title.set_position(title_position)


    for ax in [ax1, ax2]:
        ax.set_xticks(theta)
        ax.set_xticklabels(xtick_labels, fontsize=xtick_labels_font, rotation=[t * 180 / np.pi for t in theta])

        plt.gcf().canvas.draw()
        angles = np.linspace(0, 2 * np.pi, len(ax.get_xticklabels()) + 1)
        angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
        angles = np.rad2deg(angles)
        labels = []

        jj = 0
        for label, angle in zip([ax.get_xticklabels()[i] for i in right_labels],
                            [angles[i] for i in right_labels]):
            x, y = label.get_position()

            select_color = right_labels[jj]

            if len(xtick_label_colors) > 1:
                selected_color = xtick_label_colors[select_color]
            else:
               selected_color = xtick_label_colors[0]

            lab = ax.text(x, y, label.get_text(), transform=label.get_transform(), ha=label.get_ha(),
                      va=label.get_va(), fontsize=xtick_labels_font, c=selected_color)
            lab.set_rotation(angle)
            lab.set_horizontalalignment("right")
            lab.set_rotation_mode("anchor")
            labels.append(lab)
            jj += 1

        jj = 0
        for label, angle in zip([ax.get_xticklabels()[i] for i in left_labels],
                            [angles[i] for i in left_labels]):
            x, y = label.get_position()

            select_color = left_labels[jj]
            if len(xtick_label_colors) > 1:
               selected_color = xtick_label_colors[select_color]
            else:
                selected_color = xtick_label_colors[0]

            lab = ax.text(x, y, label.get_text(), transform=label.get_transform(), ha=label.get_ha(),
                      va=label.get_va(), fontsize=xtick_labels_font, c=selected_color)
            lab.set_rotation(angle)
            lab.set_horizontalalignment("left")
            lab.set_rotation_mode("anchor")
            labels.append(lab)
            jj += 1

        ax.set_xticklabels([])
        if not print_yticklabels:
            ax.set_yticklabels([])

        if rmax:
            ax.set_rmax(rmax)

    return fig
