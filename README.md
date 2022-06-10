# tl;dr

This is a **image denoising** project. ```furry-broccoli``` features:
- **image datasets**: real world noise-free and noisy images for training and learning (in ```/datasets```)
- **dataset prepper**: code for automating the task of slicing input images into tiles for machine learning
- **trained models**: trained keras models to perform image denoising (in ```/models```)
- **denoiser**: code for denoising an image given an input model

# quick start
### creating a dataset
Use ```Dataset``` to slice paired clean/noisy images into tiny tiles for training.

```python
from furrybroccoli import Dataset

image_folder = "path/to/folder"
dataset = Dataset(image_folder)

dataset.make_dataset() # this can be REALLY memory intensive
dataset.shuffle_dataset()
```
These commands will create a ```Dataset``` instance that will operate on images that are found in ```path/to/folder```.
It expects those images to have two tags in order to know which of the two is the clean one, and which is the noisy one, defaulting to
```"ISO 200"``` and ```"ISO 1600"``` respectively.

When ```make_dataset()``` is called, it will slice all image pairs into small monochrome tiles (order- and color-wise). ```shuffle_dataset()``` will randomly shuffle all tiles, keeping the correspondence between clean and noisy tiles *(useful to avoid biases when splitting the tiles into training and validation datasets)*.
The default tile size is ```28```. All parameters can be explicitly controlled and modified:

```python
dataset = Dataset(
    image_folder,
    clean_tag="clean",
    noise_tag="dirty",
    tile_size=56,
    img_type="PNG"
)
```

Stuff is contained in the ```dataset``` object, here is how to access it:

```python
# dataset folder
dataset.folder_name
'path/to/folder'

# tile size parameter
dataset.tile_size
28

# the tiles are stored in these attributes
dataset.clean_tiles_
array([[[106, 121, 168, ..., 179, 184, 168],
        [175, 100, 158, ..., 186, 130, 163],
        [186, 162, 147, ..., 128, 142, 150],
        ...,
        
        ...,
        
        ...,
        [255, 255, 255, ..., 254, 247, 251],
        [249, 239, 255, ..., 248, 255, 255],
        [255, 234, 255, ..., 236, 255, 255]]], dtype=uint8)

dataset.noise_tiles_[0].shape # the shape of each tile, also a numpy.array
(28, 28)

len(dataset.clean_tiles_)
2366595

len(dataset.noise_tiles_) # obviously they match
2366595

# did you remember to shuffle the tiles you've prepared?
dataset.dataset_shuffled
True
```

### denoising an image
Using a pretrained model and the ```Denoiser``` class.

```python
from furrybroccoli import Denoiser
from PIL import Image
import keras

image = Image.open("test_images/IMG_0053.JPG")
model = keras.models.load_model("model_56px_neuralnet_vanilla")

d = Denoiser(
    image = image,
    model = model,
    tile_size = 28 # see below for details
)
```

The ```tile_size``` parameter can be different from what the model has been trained with, but it seems to work properly this way only with ```keras``` version == ```2.6.0```. There are two ways of denoising an image, one faster than the other:

```python
d.denoise()
Denoising red channel..
width: 5184; height: 3456
Image multiple of 185×123 integer tiles.
Denoising green channel..
width: 5184; height: 3456
Image multiple of 185×123 integer tiles.
Denoising blue channel..
width: 5184; height: 3456
Image multiple of 185×123 integer tiles.
Denoised image in 'denoised_' attribute.
```

This is a first-pass denoising step that, depending on the chosen ```tile_size```, will leave artefacts at the boundaries of nearby denoised tiles. To render a perfect denoised image, it is required to perform a more thorough, multi-step denoising procedure that can be called with:

```python
d.adv_denoise()
Image properties: 5184×3456 pixels
Tile size: 28; half tile size: 14; 185×123 total tiles
Intersection Δ: 10; half Δ: 5
Pass 1/4
Denoising red channel..
width: 5184; height: 3456
Image multiple of 185×123 integer tiles.
Denoising green channel..
width: 5184; height: 3456
Image multiple of 185×123 integer tiles.
Denoising blue channel..
width: 5184; height: 3456
Image multiple of 185×123 integer tiles.
Complete. 221.7 seconds elapsed.
Pass 2/4
Denoising red channel..
width: 5184; height: 3456
Image multiple of 185×123 integer tiles.
Denoising green channel..
width: 5184; height: 3456
Image multiple of 185×123 integer tiles.
Denoising blue channel..
width: 5184; height: 3456
Image multiple of 185×123 integer tiles.
Complete. 232.1 seconds elapsed.
Pass 3/4
Denoising red channel..
width: 5170; height: 3428
Image multiple of 184×122 integer tiles.
Denoising green channel..
width: 5170; height: 3428
Image multiple of 184×122 integer tiles.
Denoising blue channel..
width: 5170; height: 3428
Image multiple of 184×122 integer tiles.
Complete. 243.0 seconds elapsed.
Pass 4/4
Denoising red channel..
width: 5184; height: 3456
Image multiple of 185×123 integer tiles.
Denoising green channel..
width: 5184; height: 3456
Image multiple of 185×123 integer tiles.
Denoising blue channel..
width: 5184; height: 3456
Image multiple of 185×123 integer tiles.
Complete. 325.9 seconds elapsed.
Reconstructing the denoised image..
```

To be continued. Note for self: call the stuff with a correct tile size (must be an EXACT mutiple)
