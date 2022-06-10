# tl;dr

This is a **image denoising** project. ```furry-broccoli``` features:
- **image datasets**: real world noise-free and noisy images for training and learning (in ```/datasets```)
- **dataset prepper**: code for automating the task of slicing input images into tiles for machine learning
- **trained models**: trained keras models to perform image denoising (in ```/models```)
- **denoiser**: code for denoising an image given an input model

# quick start
### creating a dataset
```python
from furrybroccoli import Dataset

image_folder = "path/to/folder"
dataset = Dataset(image_folder)

dataset.make_dataset() # this can be REALLY memory intensive
dataset.shuffle_dataset()
```
These commands will create a ```Dataset``` instance that will operate on images that are found in ```path/to/folder```.
It expects those images to have two tags in order to know which of the two is the clean one, and which is the noisy one, defaulting to
```"ISO 200"``` and ```"ISO 1600"``` respectively. It will keep slicing pair of images into small monochrome tiles (order- and color-wise).
Default tile size is ```28```. All parameters can be explicitly controlled and modified:

```python
dataset = Dataset(image_folder, clean_tag="clean", noise_tag="dirty", tile_size=56, img_type="PNG")
```

