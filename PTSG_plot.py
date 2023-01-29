import scipy.io
import matplotlib
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

mat = scipy.io.loadmat('data\\PTSG_sim_data_v2.mat')

x_circ = mat['x_circ']
y_circ = mat['y_circ']

plt.rcParams.update({'font.size': 22})

for n in range(0, 4):

    if n == 0:
        x_sim = mat['x_sim_a']
        y_sim = mat['y_sim_a']
        # I_emi = 0.5
        panel_name='a'
    elif n == 1:
        x_sim = mat['x_sim_b']
        y_sim = mat['y_sim_b']
        # I_emi = 0.75
        panel_name = 'b'
    elif n == 2:
        x_sim = mat['x_sim_c']
        y_sim = mat['y_sim_c']
        # I_emi = 1
        panel_name = 'c'
    elif n == 3:
        x_sim = mat['x_sim_d']
        y_sim = mat['y_sim_d']
        # I_emi = 3
        panel_name = 'd'

    fig1, ax1 = plt.subplots(1,1)
    ax1.axis("off")

    h = ax1.hist2d(x_sim[:, 0], y_sim[:, 0], bins=(100, 100), cmap='binary')
    ax1.plot(x_circ[0, :], y_circ[0, :], color='k', linewidth=2)

    fig1.colorbar(h[3], ax=ax1)

    plt.xlim([-4.1e-3, 4.1e-3])
    plt.ylim([-4.1e-3, 4.1e-3])

    ax1.set_aspect('equal', adjustable='box')

    # Create scale bar
    scalebar = ScaleBar(1, "m", length_fraction=0.25, location='lower left',
                        frameon='none')  # ,box_alpha=0)  # ,scale_loc='none')
    ax1.add_artist(scalebar)

    #plt.title('$I_e =$' + str(I_emi) + "$\,$mA")

    image_filename = 'C:\\Users\\mab679\\Pictures\\PMSI\\PTSG_plot_' + panel_name + '.png'

    #plt.savefig(image_filename, format='png', dpi=500, bbox_inches='tight')

    fig1.show()

plt.show(block=True)
