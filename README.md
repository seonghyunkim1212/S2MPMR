# Single-Shot 3D Multi-Person Shape Reconstruction from a Single RGB Image
![intro](https://user-images.githubusercontent.com/54994917/86558019-13b5ca00-bf93-11ea-8cce-774922044407.JPG)


# Dataset
You can download each dataset from the link below. 
* MuPoTS-3D [[images]](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/)[[annotations]](https://github.com/mks0601/3DMPPE_POSENET_RELEASE)
* Human3.6M [[images]](https://github.com/mks0601/3DMPPE_POSENET_RELEASE)[[annotations]](https://drive.google.com/drive/folders/189iL4dzAhaBq6TSa5NWv7Au_2heh_dsq?usp=sharing)

# Reproducing out results
You can reproduce our results with the following command.

For MuCO-3DHP dataset,
<pre>
<code>
# PCK absolute
# you should change => coord='cam'
python3 test_mupots.py 

# enter the following command
cd matlab
mpii_mupots_multiperson_eval(1,0)
</code>
</pre>

<pre>
<code>
# PCK relative and AUC relative
# you should change => coord='relative'
python3 test_mupots.py

# enter the following command
cd matlab
mpii_mupots_multiperson_eval(1,1)
</code>
</pre>

For Human3.6M dataset,
<pre>
<code>
# protocol= 'p1' or 'p2'
python3 test_h36m.py
</code>
</pre>


