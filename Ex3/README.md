How to setup the poetry environment
===

1. install python 3 (3.10.8)
2. Install pipx ``pip install pipx``
3. Install poetry ``pipx install poetry``
4. Set python 3.10 as active environment ``poetry env use python3.10``
5. (Optional) Confirm that the correct version is activated with ``poetry env list``
6. Install dependencies: Run ``poetry install`` (may take a while since torch has a lot of stuff)

How to add a package
---
* Run ``poetry add <package>``

How to run a script in poetry
---
* Run ``poetry run python3 <script.py>``

Example calls:
---
First navigate to the directory ``Ex3``
* Test with a pretrained model:
  * The test.py script first downscales the image and then uses the CNN to upscale it. Finally, it computes the PSNR score.
  * ``poetry run python3 SRCNN-pytorch-master/test.py --weights-file "SRCNN-pytorch-master/pretrained/srcnn_x4.pth" --image-file "SRCNN-pytorch-master/data/butterfly_GT.bmp" --scale 4``


* How to downscale an image separately: 
  * E.g. ``poetry run python3 src/ex3/downscale_image.py --image-file "SRCNN-pytorch-master/data/butterfly_GT.bmp" --scale 4``

NOTE
===
Currently, the way they do it is the following:
1. Downscale to a factor
2. Upscale "naive" to the original size of the image
   * I.e. just upscale the low res image to highres but keeping the "blur" so to say.
3. Upscale the so to say blurred actual resolution picture with the CNN.


Currently, they save the naive upscaled version of the picture. I think we should change that to the downscaled one.
As this would make much more sense.