# -*- coding: utf-8 -*-
# Author: Lino Grossano; lino.grossano@gmail.com
# Author: Manzini Stefano; stefano.manzini@gmail.com

__version__ = "210722"

import keras # 2.6.0
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.constraints import max_norm
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf # 2.6.0
from tensorflow.keras.optimizers import RMSprop # may be skipped

from PIL import Image
from pathlib import Path
from glob import glob
import os
from time import time


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

    Denoiser.adv_denoise() returns a denoised image, based on the model and
    input parameters, free from tiling artefacts (usually there is a visible
    artefact at each tile boundary, both horizontally and vertically).
    This is more expensive in computational time, but it allows to use smaller
    tiles which *dramatically* reduce the amount of RAM needed to process the
    entire image.
    This is usually ~4X slower than Denoiser.denoise().
    
    params
    ======
    
    image: an open PIL Image object
    model: a keras trained model object
    
    tile_size: <int> lenght, in pixel, of the square being denoised
        by the model. Higher values use higher amounts of RAM.
    
    """
    
    def __init__(self, image, model, tile_size=256, debug=False, verbose=True):
        self.image = image
        self.model = model
        self.tile_size = tile_size
        self.debug = debug
        self.verbose = verbose
        

    def _deb(self, *args, **kwargs):
        if self.debug:
            print(*args,**kwargs)

            
    def _say(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    # from Dataset class
    def _crop_in_tiles(self, image, tile_size=56, shift=0, asarray=True):

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

        assert isinstance(tile_size, int)
        assert isinstance(shift, int)
        
        self._deb(f"debug: {tile_size} tile_size in _crop_in_tiles")

        width, height = image.size

        #calculate coordinates of every tile
        for x in range (0 + shift, width, tile_size):
            if width - x < tile_size:
                continue

            for y in range (0 + shift, height, tile_size):
                if height - y < tile_size:
                    continue

                # tile coord ===
                tile_coord = (
                    x, y, # upper left coords
                    x + tile_size, y + tile_size # lower right coords
                )

                tile = image.crop(tile_coord)

                if not asarray:
                    yield tile #yielding tile as image
                else:
                    yield np.array(tile).astype("float32") / 255
        
        
    def _predict_tiles_from_image(self, image, model, tile_size=56):
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
        
        # all tiles need to be put into a list and reshaped to a form
        # the model is confortable with (x,x) to (x,x,1)
        
        self._deb(f"debug: {tile_size} tile_size in _predict_tiles_from_image")
        
        to_predict = [
            x.reshape(tile_size, tile_size, 1) for x in self._crop_in_tiles(image, tile_size)
        ]

        # the model expects an array of tiles, not a list of tiles
        to_predict = np.array(to_predict)

        return model.predict(to_predict)

    
    def _image_rebuilder(self, image, model, tile_size=56):
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
        tile_width = int(width / tile_size)
        tile_height = int(height / tile_size)

        self._say(f"Image multiple of {tile_width}×{tile_height} integer tiles.")

        for i, channel in enumerate(channels):
            
            # next line useless if we just process one channel
            #self._say(f"Processing channel {i + 1} of {len(channels)}")
            self._deb(f"debug: {tile_size} tile_size in _image_rebuilder")
            
            pred_tiles = [self._predict_tiles_from_image(channel, model, tile_size=tile_size)]

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

        
    def denoise(self, show=True, hide_extra_text=False, multichannel=False):
        #This does NOT detect multichannel mode. It's done in adv_denoise()
        
        if multichannel:

            self._say("Denoising red channel..")
            denoised_r = self._image_rebuilder(
                self.image.getchannel("R"), self.model[0], self.tile_size
            )
            
            self._say("Denoising green channel..")
            denoised_g = self._image_rebuilder(
                self.image.getchannel("G"), self.model[1], self.tile_size
            )
            
            self._say("Denoising blue channel..")
            denoised_b = self._image_rebuilder(
                self.image.getchannel("B"), self.model[2], self.tile_size
            )
            
            rgb = Image.merge("RGB",(denoised_r, denoised_g, denoised_b))
            
            self.denoised_ = rgb
            del denoised_r, denoised_g, denoised_b
            
            if not hide_extra_text: # useless if called whithin self.adv_denoise()
                self._say("Denoised image in 'denoised_' attribute.")
            
            if show:
                return rgb

        else:

            self._say("Denoising red channel..")
            denoised_r = self._image_rebuilder(
                self.image.getchannel("R"), self.model, self.tile_size
            )
            
            self._say("Denoising green channel..")
            denoised_g = self._image_rebuilder(
                self.image.getchannel("G"), self.model, self.tile_size
            )
            
            self._say("Denoising blue channel..")
            denoised_b = self._image_rebuilder(
                self.image.getchannel("B"), self.model, self.tile_size
            )
            
            rgb = Image.merge("RGB",(denoised_r, denoised_g, denoised_b))
            
            self.denoised_ = rgb
            del denoised_r, denoised_g, denoised_b
            
            if not hide_extra_text: # useless if called whithin self.adv_denoise()
                self._say("Denoised image in 'denoised_' attribute.")
            
            if show:
                return rgb



    # TODO: replace adv_denoise with this one when finished
    def adv_denoise(self, show=True, delta=10):
        """
        This function has been developed from adv_denoise().
        
        It can either denoise an image with one model for all channels,
        or using three different models to denosise red, green and blue channels
        independently.

        This function runs a 4-step denoising process that takes care to remove all
        bordering artefacts.
        This makes it possibile to use a small tile size for denoising and still
        get an artefact-free final image

        <delta>: the number of pixels to make transparent at t intersections
        """
        
        # Preparing the transparency masks

        # vertical stripes (operazione 1)
        def make_vertical_stripes(n_tiles, as_image=True, final=False):
            arrays = []

            arrays.append(np.ones((height, t - halfdelta), dtype=np.uint8)*255)
            arrays.append(np.zeros((height, delta), dtype=np.uint8))

            for _ in range(n_tiles - 2):
                arrays.append(np.ones((height, t - delta), dtype=np.uint8)*255)
                arrays.append(np.zeros((height, delta), dtype=np.uint8))        

            arrays.append(np.ones((height, t - delta), dtype=np.uint8)*255)

            if final:
                arrays.append(np.ones((height, halfdelta), dtype=np.uint8)*255) # modified
            else:
                arrays.append(np.zeros((height, halfdelta), dtype=np.uint8)) # modified

            vertical_stripes = np.concatenate(arrays, axis=1)

            if not as_image:
                return vertical_stripes
            else:
                return Image.fromarray(vertical_stripes)


        # horizontal stripes (operazione 4)
        def make_horizontal_stripes(n_tiles, as_image=True, final=False):
            arrays = []

            arrays.append(np.ones((t - halfdelta, width), dtype=np.uint8)*255)
            arrays.append(np.zeros((delta, width), dtype=np.uint8))

            for _ in range(n_tiles - 2):
                arrays.append(np.ones((t - delta, width), dtype=np.uint8)*255)
                arrays.append(np.zeros((delta, width), dtype=np.uint8))        

            arrays.append(np.ones((t - delta, width), dtype=np.uint8)*255)

            if final:
                arrays.append(np.ones((halfdelta, width), dtype=np.uint8)*255) # modified
            else:
                arrays.append(np.zeros((halfdelta, width), dtype=np.uint8))

            horizontal_stripes = np.concatenate(arrays, axis=0)

            if not as_image:
                return horizontal_stripes
            else:
                return Image.fromarray(horizontal_stripes)

        # ---
        # crops, black bars

        def make_black_img(hsize, vsize):
            channels = [np.zeros((vsize, hsize), dtype=np.uint8) for _ in range(3)]
            channels = [Image.fromarray(x) for x in channels]

            return Image.merge("RGB", channels)

        # ---

        def make_left_crop(image, amount, as_image=True):
            # amount is usually half_t

            cropshift = image.crop((amount, 0, width, image.height)) # (left, upper, right, lower)

            if not as_image:
                return np.array(cropshift)
            else:
                return cropshift


        def make_right_crop(image, amount, as_image=True):
            # amount is usually half_t

            cropshift = image.crop((0, 0, width - amount, image.height)) # (left, upper, right, lower)

            if not as_image:
                return np.array(cropshift)
            else:
                return cropshift


        def apply_left_bar(image, amount, as_image=True):
            # amount is usually half_t

            leftbar = make_black_img(amount, image.height)
            shift = np.concatenate((np.array(leftbar), np.array(image)), axis=1)

            if not as_image:
                return shift
            else:
                return Image.fromarray(shift)


        def apply_right_bar(image, amount, as_image=True):
            # amount is usually half_t

            rightbar = make_black_img(amount, image.height)

            try:
                shift = np.concatenate((np.array(image), np.array(rightbar)), axis=1)
            except:
                print("Something went wrong.")
                print(f"Image array shape: {np.array(image).shape}; bar: {np.array(rightbar).shape}")
                print("Are you trying to process a base image with alpha channel?")
                raise

            if not as_image:
                return shift
            else:
                return Image.fromarray(shift)

        # ---
        
        def make_top_crop(image, amount, as_image=True):
            # amount is usually half_t

            cropshift = image.crop((0, amount, image.width, image.height)) # (left, upper, right, lower)

            if not as_image:
                return np.array(cropshift)
            else:
                return cropshift


        def make_bottom_crop(image, amount, as_image=True):
            # amount is usually half_t

            cropshift = image.crop((0, 0, image.width, image.height - amount)) # (left, upper, right, lower)

            if not as_image:
                return np.array(cropshift)
            else:
                return cropshift


        def apply_top_bar(image, amount, as_image=True):
            # amount is usually half_t

            topbar = make_black_img(image.width, amount)
            shift = np.concatenate((np.array(topbar), np.array(image)), axis=0)

            if not as_image:
                return shift
            else:
                return Image.fromarray(shift)


        def apply_bottom_bar(image, amount, as_image=True):
            # amount is usually half_t

            bottombar = make_black_img(image.width, amount)
            shift = np.concatenate((np.array(image), np.array(bottombar)), axis=0)

            if not as_image:
                return shift
            else:
                return Image.fromarray(shift)

        # ---
        
        def apply_mask(image, mask):

            assert image.size == mask.size
            r, g, b = (image.getchannel(x) for x in "RGB")

            rgba = Image.merge("RGBA",(r, g, b, mask))

            return rgba


        def make_fixing_mask(n_tiles, as_image=True):
            """This fixes the bottom artifact in the downshift 
            image due to the denoising of black bars.

            n_tiles is the number of HORIZONTAL TILES
            """
            arrays = []
            # 0 == opaque; 255 == transparent
            # axis=0 - stack vertically
            # axis=1 - stack horizontally

            arrays.append(np.zeros((height - halfdelta, width), dtype=np.uint8))

            # bottom stripe: make this by staking stuff horizontally
            temparray = []
            temparray.append(np.ones((halfdelta, t - halfdelta), dtype=np.uint8)*255)
            for _ in range(n_tiles - 2):
                temparray.append(np.zeros((halfdelta, delta), dtype=np.uint8))
                temparray.append(np.ones((halfdelta, t - delta), dtype=np.uint8)*255)

            temparray.append(np.zeros((halfdelta, delta), dtype=np.uint8))
            temparray.append(np.ones((halfdelta, t - halfdelta), dtype=np.uint8)*255) 

            bottom_strip = np.concatenate(temparray, axis=1)

            arrays.append(bottom_strip)

            final_mask = np.concatenate(arrays, axis=0)

            if not as_image:
                return final_mask
            else:
                return Image.fromarray(final_mask)
            
        # ---
        # Let's get started
        # ---

        # INPUT
        base = self.image.convert("RGB") # don't need the eventual alpha channel
        model = self.model # already a model object OR an iterable with three models
                           # is this necessary? Nope, but it's shorter than self.model

        if isinstance(self.model, keras.engine.sequential.Sequential):
            multichannel = False
        else:
            if len(model) != 3:
                msg = "Error. <model> must contain exactly 3 models, one for each R, G, B channel."
                raise TypeError(msg)

            try:
                assert isinstance(model[0], keras.engine.sequential.Sequential)
                assert isinstance(model[1], keras.engine.sequential.Sequential)
                assert isinstance(model[2], keras.engine.sequential.Sequential)
            except AssertionError:
                print("Check the output: something's wrong with some model.")
                raise
            multichannel = True
            self._say("Multichannel mode: denoising channels with separate models.")

        width, height = base.size

        # TODO: make these selectable
        t = self.tile_size                # tile size (tile is t×t size)
        half_t = t // 2                   # half the tile size
        n_horizontal_tiles = width // t   # choose reasonably so that no pixel is lost
        n_vertical_tiles = height // t    # choose reasonably so that no pixel is lost
        #delta = 10                        # the number of pixels to make transparent at t intersections
        halfdelta = delta // 2            # the number of delta pixels in a tile

        try:
            assert width  % t == 0
            assert height % t == 0
        except AssertionError:
            print("Image width and/or height are NOT exact multiples of <tile_size>")
            print("This will not hopefully be a problem in the future, but now it is.")
            raise NotImplementedError("I can't handle non-exact multiples of tile size.")

        print(f"Image properties: {width}×{height} pixels")
        print(f"Tile size: {t}; half tile size: {half_t}; {n_horizontal_tiles}×{n_vertical_tiles} total tiles")
        print(f"Intersection Δ: {delta}; half Δ: {halfdelta}")

        # transparency masks
        vertical_mask = make_vertical_stripes(n_horizontal_tiles) # counter-intuitive
        horizontal_mask  = make_horizontal_stripes(n_vertical_tiles) # counter-intuitive

        # horizontal processing
        t0 = time()
        print("Pass 1/4")

        lc = make_left_crop(base, half_t)
        lc_rightbar = apply_right_bar(lc, half_t)
        rightshift = Denoiser(lc_rightbar, model, tile_size=self.tile_size)
        rightshift.denoise(show=False, hide_extra_text=True, multichannel=multichannel)
        rc = make_right_crop(rightshift.denoised_, half_t)
        rightshift = apply_left_bar(rc, half_t)

        t0_final = time() - t0
        print(f"Complete. {round(t0_final, 1)} seconds elapsed.")

        # vertical processing
        t1 = time()
        print("Pass 2/4")

        tc = make_top_crop(base, half_t)
        lc_bottombar = apply_bottom_bar(tc, half_t)
        downshift = Denoiser(lc_bottombar, model, tile_size=self.tile_size)
        downshift.denoise(show=False, hide_extra_text=True, multichannel=multichannel)
        bc = make_bottom_crop(downshift.denoised_, half_t)
        downshift = apply_top_bar(bc, half_t)

        t1_final = time() - t1
        print(f"Complete. {round(t1_final, 1)} seconds elapsed.")

        # diagonal processing
        t2 = time()
        print("Pass 3/4")
        diag = make_right_crop(base, half_t)
        diag = make_bottom_crop(diag, half_t)
        diag = make_left_crop(diag, half_t)
        diag = make_top_crop(diag, half_t)
        diag_d = Denoiser(diag, model, tile_size=self.tile_size)
        diag_d.denoise(show=False, hide_extra_text=True, multichannel=multichannel)
        diagshift = diag_d.denoised_
        diagshift = apply_right_bar(diagshift, half_t)
        diagshift = apply_bottom_bar(diagshift, half_t)
        diagshift = apply_top_bar(diagshift, half_t)
        diagshift = apply_left_bar(diagshift, half_t)

        t2_final = time() -t2
        print(f"Complete. {round(t2_final, 1)} seconds elapsed.")


        # Denoising the base image
        t3 = time()
        print("Pass 4/4")
        base_d = Denoiser(base, model, tile_size=self.tile_size)
        base_d.denoise(show=False, hide_extra_text=True, multichannel=multichannel)


        t3_final = time() -t3
        print(f"Complete. {round(t3_final, 1)} seconds elapsed.")


        # RECONSTRUCTION STEPS
        t4 = time()
        print("Reconstructing the denoised image..")
        # Operazione 1: rimozione delle intersezioni verticali da immagine base
        step1 = apply_mask(base_d.denoised_, vertical_mask)
        # Operazione 2: denoise dell'immagine spostata verso dx di 32 px (rappresentata in arancione):
        step2 = rightshift
        # Operazione 3: merge
        step3 = Image.alpha_composite(step2.convert("RGBA"), step1)
        # Operazione 4: rimozione delle intersezioni orizzontali da immagine base
        step4 = apply_mask(base_d.denoised_, horizontal_mask)
        # Operazione 5: denoise dell'immagine spostata verso il basso di 32 px (rappresentata in verde):
        step5 = downshift.copy()
        # Operazione 6: merge
        step6 = Image.alpha_composite(step5.convert("RGBA"), step4)
        # Operazione 7: denoise dell'immagine spostata verso dx e verso il basso di 32 px (rappresentata in rosso):
        step7 = diagshift.copy()

        final_vertical_mask = make_vertical_stripes(n_horizontal_tiles, final=True) # counter-intuitive
        final_horizontal_mask  = make_horizontal_stripes(n_vertical_tiles, final=True) # counter-intuitive

        # Operazione 8: rimozione delle intersezioni orizzontali dall'immagine da operazione 3:
        step8 = apply_mask(step3, final_horizontal_mask)
        # Operazione 9: rimozione delle intersezioni verticali dall'immagine da operazione 6:
        step9 = apply_mask(step6, final_vertical_mask)
        # Operazione 10: merge delle immagini da operazioni 7, 8 e 9:
        step10 = Image.alpha_composite(step8, step9)
        step10 = Image.alpha_composite(step7.convert("RGBA"), step10)

        # Operazione 11: applicazione maschera a step 10
        fixmask = make_fixing_mask(n_horizontal_tiles)
        step11 = apply_mask(base_d.denoised_, fixmask)

        # Operazione 12: sostituzione delle strip da immagine base denoised
        # dentro immagine de-tiled
        step12 = Image.alpha_composite(step10, step11)

        t4_final = time() -t4
        print(f"Complete. {round(t4_final, 1)} seconds elapsed.")

        tot_time = t0_final + t1_final + t2_final + t3_final + t4_final
        print(f"Total processing time: {round(tot_time, 1)} seconds.")
        
        self.detiled_ = step12.copy()
        self._say("Denoised image in 'detiled_' attribute.")
        
        # TODO anything to delete? 
        
        if show:
            return self.detiled_


# DATA PREPPING #
# ===============

class DataFeed():
    """Prepping the array takes an insane amount of RAM. In the path to
    the total conversion to generators, I'm for now using a generator here
    to bridge between the still-manageable, feature-rich (such as "dataset.shuffle()")
    Dataset objects and the vectors that the model needs.

    Taking one step further, image_width and image_height are both
    replaced by tile_size, as we're always using perfect squares.
    """

    def __init__(self, array, batch_size=128, tile_size=28, astype="float32"):
        self.array = array
        self.batch_size = batch_size
        self.tile_size = tile_size
        self.astype = astype
    
    def __iter__(self):
        for i in range(0, len(self.array), batch_size):

          if K.image_data_format() == 'channels_first':
              array_slice = self.array[i:i + self.batch_size]
              array_slice = array_slice.reshape(array_slice.shape[0], 1, tile_size, tile_size).astype(self.astype)
          else: # "channels_last"
              array_slice = self.array[i:i + self.batch_size]
              array_slice = array_slice.reshape(array_slice.shape[0], tile_size, tile_size, 1).astype(self.astype)
          
          yield array_slice / 255 # Normalize data (0-255 to 0-1)

    def __len__(self):
        return len(self.array)

        
# legacy code
def prep_array(array, img_width, img_height):
    
    """Preps an input array for the keras model. Adapted from source:

    <img_width>: the width of the tile size
    <img_height>: the height of the tile size

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


# legacy code
def get_input_shape(img_width, img_height):

    """
    <img_width>: the width of the tile size
    <img_height>: the height of the tile size

    Adapted from source:
    
    # Reshape data based on channels first / channels last strategy.
    # This is dependent on whether you use TF, Theano or CNTK as backend.
    # Source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    """
    
    if K.image_data_format() == 'channels_first':
        return (1, img_width, img_height)
    else: #"channels_last"
        return (img_width, img_height, 1)

  
# IMAGE TOOLS #
# =============

# TODO: better documentation
# TODO: put some checks
def average_img(folder, img_type="jpg", savefig=True, outfile="average.png"):
    """
    <folder> is a folder in the current working directory.
    All images that conform to type <img_type> will be blended into one (pixel average).
    
    Inspiration from:
    https://stackoverflow.com/questions/17291455/how-to-get-an-average-picture-from-100-pictures-using-pil
    """
    
    if os.path.exists(folder):
        os.chdir(folder)    # TODO: improve
        
    images = glob(f"*.{img_type}")
    
    # Assuming all images are the same size, get dimensions of first image
    w, h = Image.open(images[0]).size
    n = len(images)
    
    # Create a numpy array of floats to store the average (assume RGB images)
    arr = np.zeros((h, w, 3), np.float64)
    
    # Build up average pixel intensities, casting each image as an array of floats
    for i, img in enumerate(images):
        print(f"Processing image {i + 1} of {n}.         ", end="\r")
        arr += np.array(Image.open(img), dtype=np.float64)
    
    print("\nDone.")
    arr = arr / n
    
    print("Averaging values..")
    # Round values in array and cast as 8-bit integer
    arr = np.array(np.round(arr), dtype=np.uint8)
    
    print("Making the final image.")
    # Generate, save and preview final image
    returnimage = Image.fromarray(arr, mode="RGB")
    
    if savefig:
        print("Saving the image.")
        returnimage.save("Average.png")
    
    os.chdir("..")
    
    return returnimage


# Useful loss functions #
# =======================

def ssim(y_true, y_pred):
    """SSIM stands for Structural Similarity Index and is a perceptual metric to measure similarity of two images
    
    ideas from:
    https://stackoverflow.com/questions/57357146/use-ssim-loss-function-with-keras
    https://blog.katastros.com/a?ID=01050-ce5dc814-80fd-4fec-8d3d-1a447bcdd8c8
    
    This performs fantastically bad with respect to mse for training autoencoders for denoising
    """
    return 1 - tf.image.ssim(y_true, y_pred, max_val=1) # sputa 1 numero. quindi perchè:
    #return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0)) # ?
