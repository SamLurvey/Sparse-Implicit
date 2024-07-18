import matplotlib.pyplot as plt
import numpy as np

def plot_data(arr1, arr2):
    # Create the figure
    fig = plt.figure(figsize=(24, 8))  # Adjusted width to accommodate two plots

    # Create first subplot for the first array
    ax1 = fig.add_subplot(121, projection='3d')  # 121 means 1 row, 2 columns, first subplot

    # Create meshgrid for plotting
    x1, y1 = np.meshgrid(np.arange(arr1.shape[1]), np.arange(arr1.shape[0]))

    # Plot each layer for the first array
    for i in range(arr1.shape[2]):
        z1 = i * np.ones_like(x1)
        c1 = arr1[:,:,i]
        ax1.plot_surface(x1, y1, z1, facecolors=plt.cm.Reds(np.abs(c1)), 
                        rstride=1, cstride=1, alpha=0.3, shade=False)

    # Set labels and title for the first subplot
    ax1.set_xlabel('azimuthal')
    ax1.set_ylabel('z')
    ax1.set_zlabel('layer')
    ax1.set_title('Original Data')

    # Set axis limits for the first subplot
    ax1.set_xlim(0, arr1.shape[1])
    ax1.set_ylim(0, arr1.shape[0])
    ax1.set_zlim(0, arr1.shape[2])

    # Adjust the view angle for the first subplot
    ax1.view_init(elev=20, azim=-45)

    # Create second subplot for the second array
    ax2 = fig.add_subplot(122, projection='3d')  # 122 means 1 row, 2 columns, second subplot

    # Create meshgrid for plotting
    x2, y2 = np.meshgrid(np.arange(arr2.shape[1]), np.arange(arr2.shape[0]))

    # Plot each layer for the second array
    for i in range(arr2.shape[2]):
        z2 = i * np.ones_like(x2)
        c2 = arr2[:,:,i]
        ax2.plot_surface(x2, y2, z2, facecolors=plt.cm.Reds(np.abs(c2)), 
                        rstride=1, cstride=1, alpha=0.3, shade=False)

    # Set labels and title for the second subplot
    ax2.set_xlabel('azimuthal')
    ax2.set_ylabel('z')
    ax2.set_zlabel('layer')
    ax2.set_title('Predicted Data')

    # Set axis limits for the second subplot
    ax2.set_xlim(0, arr2.shape[1])
    ax2.set_ylim(0, arr2.shape[0])
    ax2.set_zlim(0, arr2.shape[2])

    # Adjust the view angle for the second subplot
    ax2.view_init(elev=20, azim=-45)

    plt.show()

def plot_data_efficient(arr1, arr2):
    fig = plt.figure(figsize=(24, 8))

    def plot_single(ax, arr, title):
        # Get non-zero indices
        x, y, z = np.nonzero(arr)
        
        # Create a color array
        colors = plt.cm.Reds(np.abs(arr[x, y, z]) / np.max(np.abs(arr)))
        
        # Plot using scatter
        scatter = ax.scatter(y, x, z, c=colors, alpha=0.3)
        
        ax.set_xlabel('azimuthal')
        ax.set_ylabel('z')
        ax.set_zlabel('layer')
        ax.set_title(title)
        ax.set_xlim(0, arr.shape[1])
        ax.set_ylim(0, arr.shape[0])
        ax.set_zlim(0, arr.shape[2])
        ax.view_init(elev=20, azim=-45)

    ax1 = fig.add_subplot(121, projection='3d')
    plot_single(ax1, arr1, 'Original Data')

    ax2 = fig.add_subplot(122, projection='3d')
    plot_single(ax2, arr2, 'Predicted Data')

    plt.tight_layout()
    plt.show()