import nibabel as nib
from setup import *
import numpy as np
from vedo import Volume,show,Plotter
import vedo
# print(vedo.settings)

# Print the list of default colormaps
print("Default available colormaps:")
# for cmap in default_colormaps:
    # print(cmap)
path ="test1.nii.gz"
img = nib.load(path)
data = img.get_fdata()

volume = Volume(data)


print(np.max(data))
# volume.cmap(new_cmap1, alpha =[0,1],vmin =0, vmax=0.000001)

volume.cmap("gnuplot")
# volume.alpha([0,0.01,0.1,0.2,0.3,0.8,0.85,0.9,0.95,1])
# volume.alpha([0,0.01,0.05,0.1,0.24,0.26,0.28,0,3,0.9,1])
show(volume)