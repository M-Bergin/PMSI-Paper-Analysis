import numpy as np
import math
import imageio
import matplotlib
import matplotlib.pyplot as plt
from itertools import product
import time
from scipy.io import savemat
import scipy.io
import matplotlib.animation as animation
from matplotlib_scalebar.scalebar import ScaleBar
#from pylab import *
from itertools import chain

def accum(accmap, a, func=None, size=None, fill_value=0, dtype=None):
    """
    An accumulation function similar to Matlab's `accumarray` function.

    Parameters
    ----------
    accmap : ndarray
        This is the "accumulation map_array".  It maps input (i.e. indices into
        `a`) to their destination in the output array.  The first `a.ndim`
        dimensions of `accmap` must be the same as `a.shape`.  That is,
        `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
        has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
        case `accmap[i,j]` gives the index into the output array where
        element (i,j) of `a` is to be accumulated.  If the output is, say,
        a 2D, then `accmap` must have shape (15,4,2).  The value in the
        last dimension give indices into the output array. If the output is
        1D, then the shape of `accmap` can be either (15,4) or (15,4,1)
    a : ndarray
        The input data to be accumulated.
    func : callable or None
        The accumulation function.  The function will be passed a list
        of values from `a` to be accumulated.
        If None, numpy.sum is assumed.
    size : ndarray or None
        The size of the output array.  If None, the size will be determined
        from `accmap`.
    fill_value : scalar
        The default value for elements of the output array.
    dtype : numpy data type, or None
        The data type of the output array.  If None, the data type of
        `a` is used.

    Returns
    -------
    out : ndarray
        The accumulated results.

        The shape of `out` is `size` if `size` is given.  Otherwise the
        shape is determined by the (lexicographically) largest indices of
        the output found in `accmap`.


    """

    # Check for bad arguments and handle the defaults.
    if accmap.shape[:a.ndim] != a.shape:
        raise ValueError("The initial dimensions of accmap must be the same as a.shape")
    if func is None:
        func = np.sum
    if dtype is None:
        dtype = a.dtype
    if accmap.shape == a.shape:
        accmap = np.expand_dims(accmap, -1)
    adims = tuple(range(a.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)

    # Create an array of python lists of values.
    vals = np.empty(size, dtype='O')
    for s in product(*[range(k) for k in size]):
        vals[s] = []
    for s in product(*[range(k) for k in a.shape]):
        indx = tuple(accmap[s])
        val = a[s]
        vals[indx].append(val)

    # Create the output array.
    out = np.empty(size, dtype=dtype)
    for s in product(*[range(k) for k in size]):
        if vals[s] == []:
            out[s] = fill_value
        else:
            out[s] = func(vals[s])

    return out


#files = [69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 84, 85, 86, 87, 88, 89, 92, 94]
#files_in = range(189,205)
#files_in = range(338,358)
#files_in = range(379,394)
#files_in = list(range(379, 394)) + list(range(338, 358))
files_in = 353

# Unscramble the file list
files=np.zeros_like(files_in)
N_files = 1 #len(files)
# if N_files == 1:
#     files=files_in
# else:
#     files[::2]=files_in[0:math.floor(N_files/2)]
#     files[1::2]=files_in[-1:math.floor(N_files/2)-1:-1]

# Actual check the filament value and sort
I_F_avg_scramble = ['nan']*N_files
for n_files in range(0,N_files):

    filename = "data\\SS00" + \
               str(files_in) + ".dat"

    output = np.loadtxt(filename)
    I_F_avg_scramble[n_files] = np.nanmean(output[:, 11])

# Set the order of files so the order is correct
sorted_inds = np.argsort(I_F_avg_scramble)
#files=[files_in[i] for i in sorted_inds]
files=files_in

I_FF_avg = ['nan']*N_files
I_FF_std = ['nan']*N_files
I_FF_min = ['nan']*N_files
I_FF_max = ['nan']*N_files

I_mm_total = ['nan']*N_files

I_F_avg = ['nan']*N_files

image_filename_list=[]

for n_files in range(0,N_files):

    filename = "data\\SS00" + \
               str(files) + ".dat"

    output = np.loadtxt(filename)
    file_size = output.shape

    # Plot the positions of the scan
    # plt.scatter(output[:,13], output[:,14])

    # Unique x and y values
    if file_size[1]==16:
        x_unique, x_locs = np.unique(output[:, 14], return_inverse=True)
        y_unique, y_locs = np.unique(output[:, 15], return_inverse=True)
    elif file_size[1]==15:
        x_unique, x_locs = np.unique(output[:, 13], return_inverse=True)
        y_unique, y_locs = np.unique(output[:, 14], return_inverse=True)

    map_array = np.vstack((y_locs, x_locs)).T

    # Plot the current on the ammeter

    Z_mm = -accum(map_array, output[:, 9], None, None, 'nan', None)
    Z_sh = accum(map_array, output[:, 12], None, None, 'nan', None)
    Z_L = accum(map_array, output[:, 6], None, None, 'nan', None)
    Z_tot = Z_L+Z_mm*1000

    Z_R = accum(map_array, output[:, 7], None, None, 'nan', None)
    Z_FF = accum(map_array, output[:, 8], None, None, 'nan', None)
    Z_F = accum(map_array, output[:, 11], None, None, 'nan', None)

    Z_V_sh = accum(map_array, output[:, 5], None, None, 'nan', None)

    I_F_avg[n_files] = np.nanmean(output[:, 11])
    I_FF_avg[n_files] = np.nanmean(output[:, 8])
    I_FF_std[n_files] = np.nanstd(output[:, 8])
    I_FF_min[n_files] = np.nanmin(output[:, 8])
    I_FF_max[n_files] = np.nanmax(output[:, 8])

    I_mm_total[n_files] = np.nansum(Z_mm)*((y_unique[0]-y_unique[1])**2)*1000/(1**2)

    # Remove data if it jumps very high up
    # nan_inds=Z_sh>0.1
    # Z_sh[nan_inds] = 'nan'
    # Z_tot[nan_inds] = 'nan'
    # Z_L[nan_inds] = 'nan'

    # plot_var =5.7946-Z_FF#Z_L+Z_sh+Z_mm*1000#+Z_L#+Z_FF+Z_R #+ Z_mm*1000 #+ Z_sh# + Z_FF + Z_R
    plot_var =Z_mm*1000#(5.79762510911429-Z_FF)#Z_mm*1000


    if n_files == 0:
        #fig1 = plt.figure()
        N = plot_var.shape
        fig1, ax1 = plt.subplots()
        circle = plt.Circle((np.mean(x_unique), np.mean(y_unique)), 3, color='k', fill=False,
                            linestyle='--', linewidth=2)
        ax1.add_patch(circle)
        img1 = plt.imshow(-plot_var, extent=[min(x_unique), max(x_unique), min(y_unique), max(y_unique)],
                          origin='lower', cmap='gray')  # , cmap='gray')
        plt.axis('off')
        #cbar1 = fig1.colorbar(ax1)
        cbar1 = plt.colorbar(img1, ax=ax1)
        plt.ion()
        plt.rcParams.update({'font.size': 22})
        cbar1.ax.tick_params(labelsize=22)
        image_filename='C:\\Users\\mberg\\Pictures\\PMSI\\' + filename[-11:-4] + '_v3.png'
        image_filename_list.append(image_filename)
        plt.title('$I_f =$' + "%0.2f" % I_F_avg[n_files] + "$\,$A")
        # Create scale bar
        scalebar = ScaleBar(1, "mm", length_fraction=0.25, location='lower left',
                            frameon='none')  # ,box_alpha=0)  # ,scale_loc='none')
        #ax=plt.gca()
        ax1.add_artist(scalebar)
        # ax1.set_xlim(min(x_unique)-(x_unique[1]-x_unique[0])/2, max(x_unique)+(x_unique[1]-x_unique[0])/2)
        # ax1.set_ylim(min(y_unique) - (x_unique[1]-x_unique[0])/2, max(y_unique) + (x_unique[1]-x_unique[0])/2)
        ax1.set_xlim(13.25, 16.75)
        ax1.set_ylim(17.55, 21.05)
        plt.savefig(image_filename, format='png', dpi=500, bbox_inches='tight')
        plt.show()
    else:
        img1.set_data(-plot_var)
        min_value = np.nanmin([np.nanmin(element) for element in -plot_var])
        max_value = np.nanmax([np.nanmax(element) for element in -plot_var])
        cbar1.mappable.set_clim(max_value, min_value)
        plt.title('$I_f =$' + "%0.2f" % I_F_avg[n_files] + "$\,$A")

        #image_filename = 'C:\\Users\\mberg\\Pictures\\PMSI\\' + filename[-11:-4] + '_v2.png'
        image_filename_list.append(image_filename)
        #plt.savefig(image_filename, format='png', dpi=500, bbox_inches='tight')


    fig1.canvas.flush_events()  # update the plot and take care of window events (like resizing etc.)
    time.sleep(0.5)
    # plt.savefig('C:\\Users\\mab679\\Pictures\\' + filename[-11:-4] +'_mm.png', format='png', dpi=500, bbox_inches='tight')

    # Plot the current on the liner
    # Z_L = accum(map_array, output[:, 6], None, None, 'nan', None)
    # fig2 = plt.figure()
    # ax2 = plt.imshow(Z_L, extent=[min(x_unique), max(x_unique), min(y_unique), max(y_unique)],
    #                  origin='lower')  # , cmap='gray')
    # plt.title('Liner current/mA')
    # fig2.colorbar(ax2)
    # plt.savefig('C:\\Users\\mab679\\Pictures\\' + filename[-11:-4] +'_l.png', format='png', dpi=500, bbox_inches='tight')

    # Plot both together
    # Plot the current on the liner
    # fig3 = plt.figure()
    # ax3 = plt.imshow(Z_L-Z_mm*1000, extent=[min(x_unique), max(x_unique), min(y_unique), max(y_unique)],
    #                  origin='lower')  # , cmap='gray')
    # plt.title('Total current/mA')
    # fig3.colorbar(ax3)

    # Create MATLAB files
    # mdic = {"Z_mm": Z_mm, "map_array": map_array, "output": output, 'files': files, 'I_f_avg_n': I_F_avg[n_files]}
    # savemat(filename[0:-4]+".mat", mdic)


mat = scipy.io.loadmat('data\\PSF_mksbd_v2.mat')


# Actual check the filament value and sort
I_F_avg_scramble = ['nan']*N_files
for n_files in range(0,N_files):

    filename = "data\\SS00" + \
               str(files_in) + "_X.mat"

    mat = scipy.io.loadmat(filename)

    # Create subplot
    fig, ax = plt.subplots()
    ax.axis("off")

    N=mat['J_circ'].shape

    # Plot image
    img =ax.imshow(-mat['J_circ']*1000, extent=[min(x_unique), max(x_unique), min(y_unique), max(y_unique)], origin='lower', cmap='gray')
    circle = plt.Circle((np.mean(x_unique), np.mean(y_unique)), 3, color='k', fill=False,
                        linestyle='--', linewidth=2)
    cbar1 = plt.colorbar(img, ax=ax)
    # Create scale bar
    # Create scale bar
    scalebar = ScaleBar(1, "mm", length_fraction=0.25, location='lower left',
                        frameon='none')  # ,box_alpha=0)  # ,scale_loc='none')
    ax.add_artist(scalebar)
    ax.add_patch(circle)

    ax.set_xlim(13.25, 16.75)
    ax.set_ylim(17.55, 21.05)

    plt.title('$I_f =$' + "%0.2f" % mat['I_f_avg_n'] + "$\,$A")



    # image_filename = 'C:\\Users\\mberg\\Pictures\\PMSI\\SS' + filename[-11:-4] + 'X_v2.png'
    # plt.savefig(image_filename, format='png', dpi=500, bbox_inches='tight')

    plt.show()


mat = scipy.io.loadmat('data\\PSF_mksbd_v2.mat')

plt.rcParams.update({'font.size': 40})

# Create subplot
fig, ax = plt.subplots()
ax.axis("off")

# Plot image
ax.imshow(mat['A'],origin='lower',  cmap='binary')

# Create scale bar
scalebar = ScaleBar(0.1, "mm", length_fraction=0.5,location='upper left')#,scale_loc='none')
ax.add_artist(scalebar)

# image_filename = 'C:\\Users\\mberg\\Pictures\\PMSI\\A_v3.png'

# plt.savefig(image_filename, format='png', dpi=500, bbox_inches='tight')
# image_filename = 'C:\\Users\\mberg\\Pictures\\PMSI\\A_v3.svg'
# plt.savefig(image_filename, format='svg')


plt.show(block=True)

