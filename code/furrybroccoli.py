# -*- coding: utf-8 -*-
# Author: Lino Grossano; lino.grossano@gmail.com
# Author: Manzini Stefano; stefano.manzini@gmail.com

import keras # 2.6.0
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.constraints import max_norm
from keras import backend as K
import numpy as np
import tensorflow as tf # 2.6.0
from tensorflow.keras.optimizers import RMSprop # may be skipped

from PIL import Image
from pathlib import Path
from glob import glob
import os


class Dataset:

    """Creates a Dataset object, that processes source images and cointains
       output training/validation tiles.
        
        params
        ======
        
        folder name: <str> the folder where to look for the images
        tile_size: <int>: images will be sliced into tile_size × tile_size squares
        
        clean_tag: <str>: an identifier for the clean, non-noisy images
        noise_tag: <str>: an identifier for the noisy images
        
        Note: all images should come in pair, clean and noisy, with the
        appropriate tags. See for reference:
        https://github.com/Stemanz/furry-broccoli/raw/main/datasets/standard_dataset/
        
        funcs
        =====
        
        self.make_dataset(): processes the images contained in the <folder_name>
             to squares of <tile_size> pixels.
             
             produces
             ========
             self.clean_tiles_: <np.array> with tiles from clean images 
             self.noise_tiles_: <np.array> with tiles from noisy images
             
             When an image is split into tiles, each tile is split into
             R, G, B channels and added to the growing list of tiles.
             
             Each tile is converted to a <np.array>, values untouched (0-255)      
             

        self.make_rgb_dataset(): processes the images contained in the <folder_name>
             to squares of <tile_size> pixels.
             
             produces
             ========
             self.clean_tiles_r_: <list> with tiles from the red channel of clean images
             self.clean_tiles_g_: <list> with tiles from the green channel of clean images
             self.clean_tiles_b_: <list> with tiles from the blue channel of clean images
             self.noise_tiles_r_: <list> with tiles from the red channel of noise images
             self.noise_tiles_g_: <list> with tiles from the green channel of noise images
             self.noise_tiles_b_: <list> with tiles from the blue channel of noise images
             
             Each tile is converted to a <np.array>, values untouched (0-255)
        
        
        self.shuffle_dataset(): shuffle all tiles (tile correspondences are maintained)
        
        self.shuffle_rgb_dataset(): shuffle all tiles (tile correspondences are maintained)
        
        
        """

    def __init__(self, folder_name, tile_size=28, 
                 clean_tag="ISO200", noise_tag="ISO1600",
                 img_type="JPG"
                ):
        self.folder_name = folder_name
        self.tile_size = tile_size
        
        self.dataset_shuffled = False
        self.rgb_dataset_shuffled = False
        
        # loading image names from dataset directory ===
        # for Python < 3.10 with limited glob functionality
        self.basedir = Path(os.getcwd())
        os.chdir(folder_name)
        img_files = glob(f"*.{img_type}")
        img_files = [x for x in img_files if x.endswith(f".{img_type}")]
        os.chdir(self.basedir)
        
        self.clean_pics_filenames = sorted([x for x in img_files if clean_tag in x])
        self.noise_pics_filenames = sorted([x for x in img_files if noise_tag in x])
        
        try:
            assert len(self.clean_pics_filenames) == len(self.noise_pics_filenames)
        except:
            print("**error**: mismatched length of clean and noise images lists. Details (clean/noise):")
            print(len(self.clean_pics_filenames))
            print(len(self.noise_pics_filenames))
            print(self.clean_pics_filenames)
            print(self.noise_pics_filenames)
        
        if len(self.clean_pics_filenames) == 0:
            raise TypeError(f"Are your sure the specified folders contains any suitable {img_type} image?")

    
    def _load_pic(self, image_name, folder_name):

        """
        Assumes a subdirectory <folder name> containing the
        image <image_name> to load.

        params
        ======

        image_name: <str>
        folder_name: <str>
        """

        fullpath = Path(folder_name, image_name)
        picture = Image.open(fullpath)
        return picture
    

    def _crop_in_tiles(self, image, tile_size=28, shift=0):

        """
        This function crops an image in several tiles
        tile_size × tile_size squares, yielding a tile
        every iteration.

        If the input image is not a perfect multiple of
        a(tile_size) × b(tile_size), non-square tiles are NOT
        YIELDED.

        params
        ======

        image: a Pillow open image
        tile_size: <int> pixels; size of the tile side
        shift: <int>: the offset from 0,0 in pixels
        """

        assert isinstance(tile_size, int)
        assert isinstance(shift, int)

        width, height = image.size

        #calculate coordinates of every tile
        for x in range (0+shift, width, tile_size):
            if width - x < tile_size:
                continue

            for y in range (0+shift, height, tile_size):
                if height - y < tile_size:
                    continue

                # tile coord ===
                tile_coord = (
                    x, y, # upper left coords
                    x + tile_size, y + tile_size # lower right coords
                )

                tile = image.crop(tile_coord)
                yield tile
        

    def _split_into_channels(self, image, as_array=False):
        
        if not as_array:
            return [image.getchannel(x) for x in "RGB"]
        else:
            return [np.array(image.getchannel(x)) for x in "RGB"]
    

    def make_rgb_dataset(self):
        clean_pics = (self._load_pic(x, self.folder_name) for x in self.clean_pics_filenames)
        noise_pics = (self._load_pic(x, self.folder_name) for x in self.noise_pics_filenames)

        self.clean_tiles_r_ = []
        self.clean_tiles_g_ = []
        self.clean_tiles_b_ = []
        self.noise_tiles_r_ = []
        self.noise_tiles_g_ = []
        self.noise_tiles_b_ = []
        
        for clean in clean_pics:
            tiles = self._crop_in_tiles(clean, tile_size=self.tile_size,)
            for tile in tiles:
                r,g,b = self._split_into_channels(tile, as_array=True)
                self.clean_tiles_r_.append(r)
                self.clean_tiles_g_.append(g)
                self.clean_tiles_b_.append(b)
        
        for noise in noise_pics:
            tiles = self._crop_in_tiles(noise, tile_size=self.tile_size,)
            for tile in tiles:
                r,g,b = self._split_into_channels(tile, as_array=True)
                self.noise_tiles_r_.append(r)
                self.noise_tiles_g_.append(g)
                self.noise_tiles_b_.append(b)

        # final transform of each list into a np.array
        
        self.clean_tiles_r_ = np.array(self.clean_tiles_r_)
        self.clean_tiles_g_ = np.array(self.clean_tiles_g_)
        self.clean_tiles_b_ = np.array(self.clean_tiles_b_)
        self.noise_tiles_r_ = np.array(self.noise_tiles_r_)
        self.noise_tiles_g_ = np.array(self.noise_tiles_g_)
        self.noise_tiles_b_ = np.array(self.noise_tiles_b_)
        
        
    def make_dataset(self):
        clean_pics = (self._load_pic(x, self.folder_name) for x in self.clean_pics_filenames)
        noise_pics = (self._load_pic(x, self.folder_name) for x in self.noise_pics_filenames)

        # these will store tile1_R, tile1_G, tile1_B, tile2_R, tile2_G, ..
        self.clean_tiles_ = []
        self.noise_tiles_ = []

        for clean in clean_pics:
            tiles = self._crop_in_tiles(clean, tile_size=self.tile_size,)
            for tile in tiles:
                self.clean_tiles_.extend(self._split_into_channels(tile, as_array=True))
        
        for noise in noise_pics:
            tiles = self._crop_in_tiles(noise, tile_size=self.tile_size,)
            for tile in tiles:
                self.noise_tiles_.extend(self._split_into_channels(tile, as_array=True))

        # final transform of each list into a np.array
        self.clean_tiles_ = np.array(self.clean_tiles_)
        self.noise_tiles_ = np.array(self.noise_tiles_)


    def shuffle_dataset(self):
        if hasattr(self, "clean_tiles_"):
            shuffler = np.random.permutation(len(self.clean_tiles_))
            self.clean_tiles_ = self.clean_tiles_[shuffler]
            self.noise_tiles_ = self.noise_tiles_[shuffler]
            
            self.dataset_shuffled = True
        else:
            print("Nothing to shuffle, yet. Run 'self.make_dataset()' to make one.")


    def shuffle_rgb_dataset(self):
        if hasattr(self, "clean_tiles_r_"):
            shuffler = np.random.permutation(len(self.clean_tiles_r_))
            self.clean_tiles_r_ = self.clean_tiles_r_[shuffler]
            self.clean_tiles_g_ = self.clean_tiles_g_[shuffler]
            self.clean_tiles_b_ = self.clean_tiles_b_[shuffler]
            self.noise_tiles_r_ = self.noise_tiles_r_[shuffler]
            self.noise_tiles_g_ = self.noise_tiles_g_[shuffler]
            self.noise_tiles_b_ = self.noise_tiles_b_[shuffler]
            
            self.rgb_dataset_shuffled = True
        else:
            print("Nothing to shuffle, yet. Run 'self.make_rgb_dataset()' to make one.")
        

class Denoiser():
    """ Works on PIL <Image> objects.
    Denoiser.denoise() returns a denoised image, based on the model and
    input parameters.
    
    params
    ======
    
    image: an open PIL Image object
    model: a keras trained model object
    
    tile_size: <int> lenght, in pixel, of the square being denoised
        by the model. Higher values use higher amounts of RAM.
    
    """
    
    def __init__(self, image, model,
                 tile_size, # <int> or (horizontal pixels, vertical pixels)
                 debug=False, verbose=True
    ):
        self.image = image
        self.model = model
        
        if isinstance(tile_size, tuple):
            assert len(tile_size) == 2
            self.tile_size_h, self.tile_size_v = tile_size
            assert isinstance(self.tile_size_h, (int, float))
            assert isinstance(self.tile_size_v, (int, float))
        elif isinstance(tile_size, (int, float)):
            self.tile_size_h = tile_size
            self.tile_size_v = tile_size
        else:
            raise TypeError("tile_size expected as <int> or 2-elems <tuple>")
        
        # in case floats were passed
        self.tile_size_h = int(self.tile_size_h)
        self.tile_size_v = int(self.tile_size_v)
        
        assert isinstance(debug, bool)
        assert isinstance(verbose, bool)
        
        self.debug = debug
        self.verbose = verbose
        

    def _deb(self, *args, **kwargs):
        if self.debug:
            print(*args,**kwargs)

            
    def _say(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


    # from Dataset class, *adapted* to be asymmetric
    def _crop_in_tiles(self, image, shift=0, asarray=True):

        """
        This generator function crops an image in several tiles
        tile_size × tile_size squares, yielding a tile
        every iteration.

        If the input image is not a perfect multiple of
        a(tile_size) × b(tile_size), non-square tiles are NOT
        YIELDED.

        params
        ======

        image: a Pillow open image
        tile_size: <int> pixels; size of the tile side
        shift: <int>: the offset from 0,0 in pixels
        
        yields
        ======
        
        A tile_size × tile_size np.array taken form the input image,
        but converted to float32 type and with values normalized from
        0 to 1
        """

        assert isinstance(shift, int)
        
        width, height = image.size

        #calculate coordinates of every tile
        for x in range (0 + shift, width, self.tile_size_h):
            if width - x < self.tile_size_h:
                continue

            for y in range (0 + shift, height, self.tile_size_v):
                if height - y < self.tile_size_v:
                    continue

                # tile coord ===
                tile_coord = (
                    x, y, # upper left coords
                    x + self.tile_size_h, y + self.tile_size_v # lower right coords
                )

                tile = image.crop(tile_coord)

                if not asarray:
                    yield tile #yielding tile as image
                else:
                    yield np.array(tile).astype("float32") / 255
        
        
    def _predict_tiles_from_image(self, image, model):
        """ This gives back the denoised <tiles>, according to the loaded <model>
        The model operates on multiple tiles at once. All tiles are shaped into a form
        that the model was trained for, then all put into a np.array container.
        This is the way the models expects the tiles for the prediction.

        NOTE: This function relies on crop_in_tiles() function.

        params
        ======

        image: a pillow Image object
        model: a keras trained model
        
        tile_size: <int> pixels. The model will operate and predict on a square with
            <tile_size> side. Higher values allocate higher amounts of RAM.
            
        returns
        ======
        
        A np.array containing all denoised (predicted) tiles
        """
                
        to_predict = [
            x.reshape(self.tile_size_v, self.tile_size_h, 1) for x in self._crop_in_tiles(image)
        ]

        # the model expects an array of tiles, not a list of tiles
        to_predict = np.array(to_predict)

        return model.predict(to_predict)

    
    def _image_rebuilder(self, image, model):
        """ Takes as input a monochromatic (single-channel) image,
        returns a denoised monochromatic image.
        
        params
        ======
        
        image: a PIL monochromatic image. ONLY ONE channel is supported
        model: a trained keras model to denoise the input image
        
        tile_size: <int> pixels of a square side to process at once.
            This is not related to the tile_size the model has been built
            with, but dictates how big is the square the model is fed with
            for denoising. The bigger this parameter, the more RAM is needed
            to perform the denoising.
            This cannot be higher than the image phisical size.
            
        returns
        =======
        A monochromatic, denoised PIL.Image object
        
        """

        # I was initially wondering to manage the channel splitting here,
        # but as the model is currently working on monochromatic images,
        # and will eventually manage the three channels with three different
        # models (again, with one channel per image), this stub of implementation
        # is not necessary anymore.
        # TODO: clear the clutter
        channels = [image]

        width, height = channels[0].size #all three channels have the same size
        self._say(f"width: {width}; height: {height}")

        # TODO
        # for now, we support only exact multiples of tile_size
        tile_width = int(width / self.tile_size_h)
        tile_height = int(height / self.tile_size_v)

        self._say(f"Image multiple of {tile_width}×{tile_height} integer tiles.")

        for i, channel in enumerate(channels):
            
            # next line useless if we just process one channel
            #self._say(f"Processing channel {i + 1} of {len(channels)}")            
            pred_tiles = [self._predict_tiles_from_image(channel, model)]

            self._deb(f"Predicted tiles length: {len(pred_tiles[0])}")

            # now we need to rebuild a numpy array based on the tile_width*tile_height original geometry        
            gen = (x for x in pred_tiles[0])

            # the final assembly is very fast ===
            returnimage = []

            #for i in range(tile_height):
            #    row_tiles = next(gen)
            #    for j in range(tile_width - 1):
            #        next_tile = next(gen)
            #        row_tiles = np.concatenate((row_tiles, next_tile), axis=1)
            #    returnimage.append(row_tiles)
            #
            #returnimage = np.array(returnimage)
            #returnimage = np.vstack(returnimage)

            for i in range(tile_width):
                row_tiles = next(gen)
                for j in range(tile_height - 1):
                    next_tile = next(gen)
                    row_tiles = np.concatenate((row_tiles, next_tile), axis=0)
                returnimage.append(row_tiles)

            returnimage = np.array(returnimage)
            returnimage = np.hstack(returnimage)

            # from array (0-1) to Image (0-255)
            returnimage = np.uint8(returnimage * 255)
            
            # discarding the last dimension
            return Image.fromarray(returnimage[:,:,0])   

        
    def denoise(self):
        
        self._say("Denoising red channel..")
        denoised_r = self._image_rebuilder(
            self.image.getchannel("R"), self.model
        )
        
        self._say("Denoising green channel..")
        denoised_g = self._image_rebuilder(
            self.image.getchannel("G"), self.model
        )
        
        self._say("Denoising blue channel..")
        denoised_b = self._image_rebuilder(
            self.image.getchannel("B"), self.model
        )
        
        rgb = Image.merge("RGB",(denoised_r, denoised_g, denoised_b))
        
        
        self.denoised_ = rgb
        del denoised_r, denoised_g, denoised_b
        self._say("Denoised image in 'denoised_' attribute.")
        
        return rgb

    
    def denoise_monochrome(self):
        """This assumes the input image only has one channel.
        """
        
        self._say("Denoising input image..")
        denoised = self._image_rebuilder(
            self.image, self.model
        )
        
        return denoised


def prep_array(array
              # img_width, img_height
              ):
    
    global img_width, img_height
    """Preps an input array for the keras model. Adapted from source:

    Reshape data based on channels first / channels last strategy.
    This is dependent on whether you use TF, Theano or CNTK as backend.
    Source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    """
    
    if K.image_data_format() == 'channels_first':
        array = array.reshape(array.shape[0], 1, img_width, img_height)
    else: #"channels_last"
        array = array.reshape(array.shape[0], img_width, img_height, 1)

    # e se provassi..? numpy.half / numpy.float16
    array = array.astype('float32')
    return array / 255 # Normalize data (0-255 to 0-1)


def get_input_shape():

    # Adapted from source:
    
    # Reshape data based on channels first / channels last strategy.
    # This is dependent on whether you use TF, Theano or CNTK as backend.
    # Source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    
    global img_width, img_height
    
    if K.image_data_format() == 'channels_first':
        return (1, img_width, img_height)
    else: #"channels_last"
        return (img_width, img_height, 1)