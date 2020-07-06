# Single-Shot 3D Multi-Person Shape Reconstruction from a Single RGB Image



# Dataset
You can download each dataset from the link below. 
* MuCo-3DHP [[data]](https://github.com/mks0601/3DMPPE_POSENET_RELEASE)
* MuPoTS-3D [[images]](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/)[[annotations]](https://github.com/mks0601/3DMPPE_POSENET_RELEASE)
* MS-COCO [[data]](https://cocodataset.org/#home)

# Training
You can reproduce our results with the following command.

For MuCO-3DHP dataset,
<pre>
<code>
# PCK absolute
./run_muco_cam.sh

# after trainig, enter the following command
cd matlab
mpii_mupots_eval(1,0)
</code>
</pre>

<pre>
<code>
# PCK relative and AUC relative
./run_muco_relative.sh

# enter the following command
cd matlab
mpii_mupots_eval(1,1)
</code>
</pre>


