import matplotlib.pylab as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import pandas as pd
import numpy as np
from cell import analysis
import utils

pd.options.display.float_format = '{:,.4f}'.format


def plot_3D(xyz, **kwargs):

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
    plot_size = kwargs.get('plot_size', (10, 10))
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

    fg = sns.FacetGrid(data, col=groupby_col, col_wrap=col_wrap, height=height)
    fg.map_dataframe(draw_heatmap, column, index, value, cbar=False, square=True)

    for ax in fg.axes.flat:
        ax.set_aspect('equal', 'box')
    plt.show()


def Grid_Heatmap(my_dict):
    """
    Takes a dictionary of data frames and plot their heatmap in a grid,
    For each data fram group it based on a column and plot a heatmap

    Parameters
    ----------
    data: a dictionary of data frames
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
    df = my_dict.copy()
    data = pd.DataFrame()

    for k, v in df.items():
        v = v.stack().reset_index()
        v["channel_id"] = k
        v.columns = ["index", "column", "value", "channel_id"]
        data = data.append(pd.DataFrame(v), ignore_index=True)
    print(data.index)

    Facet_Grid_Heatmap(data, "channel_id", len(data.keys()), 5, "index", "column", "value")
    return data


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


def plot_ER(emitter, receiver, figsize, plot_dim, annotation,  resolution, annotation_label=None, E_color=None,\
            R_color=None, E_marker='o', R_marker='x', xlim=None, ylim=None, zlim=None, E_sublist_to_color=None, \
            R_sublist_to_color=None, side_by_side=False, scatter_point_size=None):

    """
    plot right and left embeddings on one plot
    parameters
    ----------
    E: df of emitter embeddings
    R: df of receiver embeddings
    figsize: figsize
    plot_dim: 2d or 3d plot
    annotation: if True, it will use the index of E and R for embedding
    E_color: color of emitter points
    R_color: color of receiver points
    E_marker: E_marker
    R_marker: R_marker
    use_type_colors: if True, it will use "cluster_color" of each df for colors
    xlim: tuple of x limits
    ylim: tuple of y limits
    E_sublist_to_color: plot every point but color only this sublist from E
    R_sublist_to_color: plot every point but color only this sublist from R
    side_by_side: if True, plot E and R in two separate plot side by side otherwise it will plot both in the same plot

    """
    if scatter_point_size is None:
        scatter_point_size = 30
    E = emitter.copy()
    R = receiver.copy()

    E['print_index'] = True
    R['print_index'] = True

    if resolution == "cluster_label":
        resolution_color = "cluster_color"

    if resolution == "subclass_label":
        resolution_color = "subclass_color"

    if resolution == "class_label":
        resolution_color = "class_color"

    if E_sublist_to_color is not None:
        if R_sublist_to_color is not None:
            colorful_idx = E_sublist_to_color.index.tolist()
            E.loc[[i for i in E.index.tolist() if i not in colorful_idx], resolution_color] = "#D3D3D3"
            E.loc[[i for i in E.index.tolist() if i not in colorful_idx], 'print_index'] = False

            colorful_idx = R_sublist_to_color.index.tolist()
            R.loc[[i for i in R.index.tolist() if i not in colorful_idx], resolution_color] = "#D3D3D3"
            R.loc[[i for i in R.index.tolist() if i not in colorful_idx], 'print_index'] = False

    if E_sublist_to_color is not None:
        if R_sublist_to_color is None:
            colorful_idx = E_sublist_to_color.index.tolist()
            E.loc[[i for i in E.index.tolist() if i not in colorful_idx], resolution_color] = "#D3D3D3"
            E.loc[[i for i in E.index.tolist() if i not in colorful_idx], 'print_index'] = False
            R[resolution_color] = "#D3D3D3"
            R['print_index'] = False

    if E_sublist_to_color is None:
        if R_sublist_to_color is not None:
            colorful_idx = R_sublist_to_color.index.tolist()
            R.loc[[i for i in R.index.tolist() if i not in colorful_idx], resolution_color] = "#D3D3D3"
            R.loc[[i for i in R.index.tolist() if i not in colorful_idx], 'print_index'] = False
            E[resolution_color] = "#D3D3D3"
            E['print_index'] = False


    if E_color is None:
        E_color = E[resolution_color]
        R_color = R[resolution_color]


    fig = plt.figure(figsize=figsize)

    data = pd.concat((E, R))

    lim1 = np.floor(np.min(pd.concat((data['Z0'], data['Z1']))))
    lim2 = np.ceil(np.max(pd.concat((data['Z0'], data['Z1']))))

    def print_annotation(annotation, ax, data, plot_dim):
        data = data[data['print_index']==True]
        if annotation:
            if annotation_label is not None:
                annot = data[annotation_label].tolist()
            else:
                annot = data.index.tolist()
            if plot_dim == 3:
                for j, txt in enumerate(annot):
                    ax.text(data['Z0'][j], data["Z1"][j], data['Z2'][j], txt, size=10)
            else:
                for j, txt in enumerate(annot):
                    ax.text(data['Z0'][j], data["Z1"][j], txt, size=10)



    def set_axis_lim(plot_dim, xlim, ylim, zlim):

        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(lim1, lim2)

        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(lim1, lim2)

        if plot_dim==3:
            if zlim is not None:
                ax.set_zlim(zlim)
            else:
                ax.set_zlim(lim1, lim2)

    if plot_dim == 3:
        if side_by_side:
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(E['Z0'], E['Z1'], E['Z2'], c=E_color, s=scatter_point_size, marker=E_marker, label='E', alpha=1)
            print_annotation(annotation, ax, E, plot_dim)
            set_axis_lim(plot_dim, xlim, ylim, zlim)

            ax = fig.add_subplot(122, projection='3d')
            ax.scatter(R['Z0'], R['Z1'], R['Z2'], c=R_color, s=scatter_point_size, marker=R_marker, label='R', alpha=1)
            print_annotation(annotation, ax, R, plot_dim)
            set_axis_lim(plot_dim, xlim, ylim, zlim)

        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(E['Z0'], E['Z1'], E['Z2'], c=E_color, s=scatter_point_size, marker=E_marker, label='E', alpha=1)
            ax.scatter(R['Z0'], R['Z1'], R['Z2'], c=R_color, s=scatter_point_size, marker=R_marker, label='R', alpha=1)
            print_annotation(annotation, ax, data, plot_dim)
            set_axis_lim(plot_dim, xlim, ylim, zlim)

    else:
        if side_by_side:
            ax = fig.add_subplot(1, 2, 1)
            ax.scatter(E["Z0"], E["Z1"], c=E_color, s=scatter_point_size, marker=E_marker, label="E", alpha=1)
            print_annotation(annotation, ax, E, plot_dim)
            set_axis_lim(plot_dim, xlim, ylim, zlim)


            ax = fig.add_subplot(1, 2, 2)
            ax.scatter(R["Z0"], R["Z1"], c=R_color, s=scatter_point_size, marker=R_marker, label="R", alpha=1)
            print_annotation(annotation, ax, R, plot_dim)
            set_axis_lim(plot_dim, xlim, ylim, zlim)

        else:
            ax = fig.add_subplot(111)
            ax.scatter(E['Z0'], E['Z1'], c=E_color, s=scatter_point_size, marker=E_marker, label="E", alpha=1)
            ax.scatter(R['Z0'], R['Z1'], c=R_color, s=scatter_point_size, marker=R_marker, label="R", alpha=1)
            print_annotation(annotation, ax, data, plot_dim)
            set_axis_lim(plot_dim, xlim, ylim, zlim)



    plt.legend()
    plt.show()
    fig.savefig("/Users/fahimehb/Documents/NPP_GNN_project/dat/fig/2d_umap_emb.pdf")


def plot_multiple_dict(mydict, xlabel="epochs", ylabel="nandcg", x_label_rotation=90, order_of_x_values=None):
        """
        take a dictionary in which each value is a dictionary itself and plot each of these values

        Args:
        _____
        mydict: a dictinary that each value is a dictinary itself and we want to plot these dictinaries
        order_of_x_values: list, if given, it will plot with that order

        """
        for k, v in mydict.items():
            if order_of_x_values:
                xvals = order_of_x_values
                yvals = [v[i] for i in xvals]
            else:
                xvals = v.keys()
                yvals = [v[i] for i in xvals]
            plt.scatter([i for i in range(len(xvals))], yvals)
            plt.plot([v for v in yvals], label=k)
            plt.xticks([i for i in range(len(xvals))], xvals, rotation=x_label_rotation)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()

        plt.show()





def plot_node_average_ndcg(adj, e_to_r, figsize, k):
    '''
    Takes the true relevance and the distance between emitter and receiver matrix and
    plot the node average ndcg, see also Compute_node_average_ndcg for more explanation
    '''

    fig = plt.figure(figsize=figsize)
    nodes = [j for j in e_to_r.index.tolist()]
    nandcg = {}
    for n in nodes:

        _, _, _, nandcg[n] = analysis.get_distance_ndcg_score(n, e_to_r, adj, k=k)

    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(nandcg.keys()))], nandcg.values())
    ax.plot([v for v in nandcg.values()])
    ax.set_xticks([i for i in range(len(nandcg.keys()))])
    ax.set_xticklabels(nandcg.keys(), rotation=90)
    ax.set_xlabel("node_id")
    ylab = "ndcg@" + str(k)
    ax.set_ylabel(ylab)
    plt.show()
    return nandcg, np.mean([v for v in nandcg.values()])


def plot_closest_nodes(node_id, E, R, adj, topn, n_emb, node_act, cldf=None, resolution=None, plot_dim=2,
                       scatter_point_size=50, alpha=1, plot_size=(10, 10), annotation=False, theta1=30,
                       theta2=30, xlim=None, ylim=None, zlim=None):
    '''
    Plot the closest neighbors of a node with annotation and color

    Args:
    -----
    node_id: string, the node id that we want the neighbors to be shown
    E: emitter representations
    R: receiver representations
    adj: adjacency matrix
    topn: topn neighbors to be shown
    n_emb: embedding size
    node_act: "E" or "R", if the query node is emitter or reciver
    cldf: the reference table to read the metadata
    resolution: "cluster_label", "class_label", or "subclass_label"

    return:
    -------
    The emitter and receiver dataframe and plot
    '''

    df_columns_coor = ["Z" + str(i) for i in range(n_emb)]

    e_to_r_dist = analysis.get_distance_between_eachrow_of_one_df_with_all_rows_of_other_df(E[df_columns_coor],
                                                                                            R[df_columns_coor])
    info = analysis.get_closest_nodes_info(node_id, e_to_r_dist, adj, topn, cldf, resolution, node_act)
    nn_index = info.predicted_closest_neighbors_index.tolist()


    if resolution is not None:

        if cldf is None:
            utils.raise_error("When resolution is given, cldf is required")
        else:
            ref = cldf.copy()
            ref = ref.reset_index()
            ref['cluster_id'] = ref['cluster_id'].apply(str)
            ref['subclass_id'] = ref['subclass_id'].apply(str)
            ref['class_id'] = ref['class_id'].apply(str)

        if resolution == "cluster_label":
            resolution_id = "cluster_id"
            resolution_color = "cluster_color"

        if resolution == "subclass_label":
            resolution_id = "subclass_id"
            resolution_color = "subclass_color"

        if resolution == "class_label":
            resolution_id = "class_id"
            resolution_color = "class_color"

        df_columns = df_columns_coor + [
            resolution_id, resolution, resolution_color]

    if node_act == "E":
        emi_df = pd.DataFrame(index=[node_id], columns=df_columns)
        rec_df = pd.DataFrame(index=nn_index, columns=df_columns)
    else:
        rec_df = pd.DataFrame(index=[node_id], columns=df_columns)
        emi_df = pd.DataFrame(index=nn_index, columns=df_columns)

    for i in emi_df.index.tolist():
        emi_df.loc[i] = E.loc[i][df_columns_coor]
        emi_df.loc[i][resolution] = ref[ref[resolution_id] == i][resolution].tolist()[0]
        emi_df.loc[i][resolution_id] = ref[ref[resolution_id] == i][resolution_id].tolist()[0]
        emi_df.loc[i][resolution_color] = ref[ref[resolution_id] == i][resolution_color].tolist()[0]

    emi_df[df_columns_coor] = emi_df[df_columns_coor].astype(float)

    for i in rec_df.index.tolist():
        rec_df.loc[i] = R.loc[i][df_columns_coor]
        rec_df.loc[i][resolution] = ref[ref[resolution_id] == i][resolution].tolist()[0]
        rec_df.loc[i][resolution_id] = ref[ref[resolution_id] == i][resolution_id].tolist()[0]
        rec_df.loc[i][resolution_color] = ref[ref[resolution_id] == i][resolution_color].tolist()[0]

    rec_df[df_columns_coor] = rec_df[df_columns_coor].astype(float)

    fig = plt.figure(figsize=plot_size)

    if plot_dim == 2:
        ax = fig.add_subplot(111)
        ax.scatter(emi_df['Z0'], emi_df['Z1'], color=emi_df[resolution_color], s=scatter_point_size, alpha=alpha,
                   marker='o')
        ax.scatter(rec_df['Z0'], rec_df['Z1'], color=rec_df[resolution_color], s=scatter_point_size, alpha=alpha,
                   marker='x')

        if annotation:
            for j, txt in enumerate(emi_df[resolution].tolist()):
                ax.text(emi_df['Z0'][j], emi_df["Z1"][j], txt, size=10)
            for j, txt in enumerate(rec_df[resolution].tolist()):
                ax.text(rec_df['Z0'][j], rec_df["Z1"][j], txt, size=10)

    if plot_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(emi_df['Z0'], emi_df['Z1'], emi_df["Z2"], c=emi_df[resolution_color], s=scatter_point_size,
                   alpha=alpha, marker='o')
        ax.scatter(rec_df['Z0'], rec_df['Z1'], rec_df["Z2"], c=rec_df[resolution_color], s=scatter_point_size,
                   alpha=alpha, marker='x')

        if annotation:
            for j, txt in enumerate(emi_df[resolution].tolist()):
                ax.text(emi_df['Z0'][j], emi_df["Z1"][j], emi_df["Z2"][j], txt, size=10)
            for j, txt in enumerate(rec_df[resolution].tolist()):
                ax.text(rec_df['Z0'][j], rec_df["Z1"][j], rec_df["Z2"][j], txt, size=10)

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

    rows = emi_df[resolution_id].tolist()
    rows_names = emi_df[resolution].tolist()
    cols = rec_df[resolution_id].tolist()
    cols_names = rec_df[resolution].tolist()
    sub_adj = adj.loc[rows][cols]
    sub_adj.index = rows_names
    sub_adj.columns = cols_names
    if node_act == "E":
        sub_adj = sub_adj.T
        print(sub_adj)
    else:
        print(sub_adj)

    return emi_df, rec_df

