import matplotlib.pylab as plt

def Plot_3D(xyz, colors, annotation, theta1, theta2, **kwargs):
    """
    Plot a rotateable 3d plot

    parameters
    ----------
    xyz: a np.array with a shape of (n, 3)
    color: a list of colors with the correct order of the shape n
    annotation: a list of annotation with correct order of shape n
    theta1 : first rotation angle
    theta2 : second rotation angle
    ax_labels : list of axis labels
    plot_size: tuple of plot sizes

    """
    plot_size = kwargs.get('plot_size', None)
    if plot_size is not None:
        fig = plt.figure(figsize=plot_size)
    else:
        fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    ax.scatter(x, y, z, color=colors, s=100)
    if annotation is not None:
        for i, txt in enumerate(annotation):
            ax.text(x[i,], y[i], z[i], txt, size=10)

    ax_labels = kwargs.get('ax_labels', None)
    if ax_labels is not None:
        ax.set_xlabel(ax_labels[0], fontsize=20)
        ax.set_ylabel(ax_labels[1], fontsize=20)
        ax.set_zlabel(ax_labels[2], fontsize=20)

    if theta1 and theta2 is not None:
        ax.view_init(theta1, theta2)

    return ax


