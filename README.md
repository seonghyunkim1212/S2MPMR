# Single-Shot 3D Multi-Person Shape Reconstruction from a Single RGB Image
![intro](https://user-images.githubusercontent.com/54994917/86558019-13b5ca00-bf93-11ea-8cce-774922044407.JPG)


# Dataset
You can download each dataset from the link below. 
* MuCo-3DHP [[data]](https://github.com/mks0601/3DMPPE_POSENET_RELEASE)
* MuPoTS-3D [[images]](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/)[[annotations]](https://github.com/mks0601/3DMPPE_POSENET_RELEASE)
* MS-COCO [[data]](https://cocodataset.org/#home)
* Human3.6M [[images]](https://github.com/mks0601/3DMPPE_POSENET_RELEASE)[[annotations]](https://drive.google.com/drive/folders/189iL4dzAhaBq6TSa5NWv7Au_2heh_dsq?usp=sharing)
* MPII [[images]](http://human-pose.mpi-inf.mpg.de/)[[annotations]](https://drive.google.com/file/d/1LiCTqfe0A0kdvNpd8YUVQrreqevfL14d/view?usp=sharing)
* Mosh [[data]](https://drive.google.com/file/d/1c0R20yQv-HXhKW7mVmguC8Nfz9IPak-_/view?usp=sharing)

# Training
You can reproduce our results with the following command.

For MuCO-3DHP dataset,
<pre>
<code>
# PCK absolute
./run_muco_cam.sh

# after trainig, enter the following command
cd matlab
mpii_mupots_multiperson_eval(1,0)
</code>
</pre>

<pre>
<code>
# PCK relative and AUC relative
./run_muco_relative.sh

# enter the following command
cd matlab
mpii_mupots_multiperson_eval(1,1)
</code>
</pre>

For Human3.6M dataset,
<pre>
<code>
./run_h36m.sh
</code>
</pre>


