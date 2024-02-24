How to setup the poetry environment
===

1. install python 3 (3.10.8)
2. Install pipx: ``pip install pipx``
3. Install poetry: ``pipx install poetry``
4. Set python 3.10 as active environment: ``poetry env use python3.10``
5. (Optional) Confirm that the correct version is activated with: ``poetry env list``
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
  * The ``test.py`` script first downscales the image and then uses the CNN to upscale it. Lastly, it computes the PSNR score.
  * E.g. ``poetry run python3 SRCNN-pytorch-master/test.py --weights-file "SRCNN-pytorch-master/pretrained/srcnn_x4.pth" --image-file "SRCNN-pytorch-master/data/butterfly_GT.bmp" --scale 4``


* How to downscale an image separately:
  * E.g. ``poetry run python3 src/ex3/downscale_image.py --image-file "SRCNN-pytorch-master/data/butterfly_GT.bmp" --scale 4``


How to prepare
---
Example: `` poetry run python3 SRCNN-pytorch-master/prepare.py --images-dir train2014_2k/ --output-path train2014_2k.h5 --scale 3``


How to learn
---
Example: ````


NOTE
===
The way it is done is the following:
1. Downscale to a factor F (e.g. 4)
2. Save the downscaled version as [...]_bicubic_x<F>.bmp
3. Upscale again to the original size of the image
   * I.e. just upscale the low res image to highres but not adding any information.
4. Add information to the image with the CNN. (Now it is really upscaled).
