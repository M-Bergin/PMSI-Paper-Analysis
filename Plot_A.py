import scipy.io
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import matplotlib.pyplot as plt


mat = scipy.io.loadmat('PSF_mksbd_v2.mat')

plt.rcParams.update({'font.size': 22})

# Create subplot
fig, ax = plt.subplots()
ax.axis("off")

# Plot image
ax.imshow(mat['A'],origin='lower',  cmap='binary')

# Create scale bar
scalebar = ScaleBar(0.1, "mm", length_fraction=0.5,location='upper left')#,scale_loc='none')
ax.add_artist(scalebar)

image_filename = 'C:\\Users\\mberg\\Pictures\\PMSI\\A_v2.png'

# plt.savefig(image_filename, format='png', dpi=500, bbox_inches='tight')


plt.show()

files_in = range(338,358)
#files_in = range(379,394)

# Unscramble the file list
files=np.zeros_like(files_in)
N_files = len(files)
# if N_files == 1:
#     files=files_in
# else:
#     files[::2]=files_in[0:math.floor(N_files/2)]
#     files[1::2]=files_in[-1:math.floor(N_files/2)-1:-1]

# Actual check the filament value and sort
I_F_avg_scramble = ['nan']*N_files
for n_files in range(0,N_files):

    filename = "data\\SS00" + \
               str(files_in[n_files]) + "_X.mat"

    mat = scipy.io.loadmat(filename)

    # Create subplot
    fig, ax = plt.subplots()
    ax.axis("off")

    N=mat['J_circ'].shape

    # Plot image
    img =ax.imshow(-mat['J_circ']*1000, origin='lower', cmap='gray')
    circle = plt.Circle((np.floor(N[0]/2), np.floor(N[1]/2)), np.floor(N[0]/2), color='k', fill=False, linestyle='--',linewidth=2)
    cbar1 = plt.colorbar(img, ax=ax)
    # Create scale bar
    scalebar = ScaleBar(0.1, "mm", length_fraction=0.25, location='lower left',frameon='none')#,box_alpha=0)  # ,scale_loc='none')
    ax.add_artist(scalebar)
    ax.add_patch(circle)

    plt.title('$I_f =$' + "%0.2f" % mat['I_f_avg_n'] + "$\,$A")



    image_filename = 'C:\\Users\\mberg\\Pictures\\PMSI\\SS' + filename[-11:-4] + '_v2.png'

    # plt.savefig(image_filename, format='png', dpi=500, bbox_inches='tight')

    plt.show()



