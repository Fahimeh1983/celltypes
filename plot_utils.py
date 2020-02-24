import matplotlib.pylab as plt
import seaborn as sns

def Plot_3D(xyz, colors, annotation, theta1, theta2, **kwargs):

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

    plot_size = kwargs.get('plot_size', (20, 20))
    plot_title = kwargs.get('plot_title', None)
    x_label = kwargs.get('x_label', "x")

    fig = plt.figure(figsize= plot_size)
    ax = fig.add_subplot(111, projection='3d')
    x = xyz["Z0"]
    y = xyz["Z1"]
    z = xyz["Z2"]

    if 'cluster_color' not in xyz:
        c = "Blue"
    else:
        c = xyz["cluster_color"]

    ax.scatter(x, y, z, color=c, s=100)
    if annotation:
        for i, txt in enumerate(xyz.index.tolist()):
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

    return fig


def Scatter_plot(datasets, datasets_colors, datasets_legends, **kwargs):

    """
    Takes multiple datasets and scatter plot each of them

    Parameters
    ----------
    figsize: size of the figure
    datasets: a dict of dataframes, each of the dataframes going to be a scatter plot
    datasets_colors: a dict of colors (one color per dataframe)
    datasets_legends: a dict of legends (one legend per dataframe)

    Optional parameters
    --------------------
    figsize:  a tuple of fig size
    x_label: label of x axis
    y_label: label of y axis
    xticks: a list of numbers for the xtick locations
    xtick_labels: a list of xtick labels
    ytick: a list of numbers for the yticks locations
    ytick_labels : a list of y tick labels
    xticl_label_rotation: rotation angel for xtick labels
    ytick_label_rotation: rotation angle for ytick labels
    legend_loc : location of legend
    """

    figsize = kwargs.get('figsize', (10, 10))
    x_label = kwargs.get('x_label', "x")
    y_label = kwargs.get('y_label', "y")
    legend_loc = kwargs.get('legend_loc', "upper left")
    xtick_labels = kwargs.get('xtick_labels', None)
    ytick_labels = kwargs.get('ytick_labels', None)
    xtick_label_rotation = kwargs.get('xtick_label_rotation', None)
    ytick_label_rotation = kwargs.get('ytick_label_rotation', None)
    plot_line = kwargs.get('plot_line', True)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for k, v in datasets.items():
        ax.scatter(v['x'], v['y'], s=10, c=datasets_colors[k], label=datasets_legends[k])
        if plot_line:
            ax.plot(v['x'], v['y'], c=datasets_colors[k])

    plt.legend(loc=legend_loc)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    if xtick_labels is not None:
        xticks = kwargs.get('xticks', None)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=xtick_label_rotation)

    if ytick_labels is not None:
        yticks = kwargs.get('yticks', None)
        ax.set_xticks(yticks)
        ax.set_xticklabels(ytick_labels, rotation=ytick_label_rotation)

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
    Takes a data fram, group it based on a column and plot a heatmap for each group

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
