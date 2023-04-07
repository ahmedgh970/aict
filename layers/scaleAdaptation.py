# ==============================================================================
# MIT License

# Copyright (c) 2023 Ahmed Ghorbel

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
Modified from: https://github.com/treammm/ResizeCompression
"""

import sys
import os
import tensorflow as tf
from .ConvNeXt import *

def get_pixel_value(img, x, y):
    """
    Copied from: https://github.com/kevinzakka/spatial-transformer-network
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)

    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def affine_grid_generator(height, width, theta, num_batch):
    """
    Modified from: https://github.com/kevinzakka/spatial-transformer-network
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.

    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.

    - width: desired width of grid/output. Used
      to downsample or upsample.

    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.

    Returns
    -------
    - unnormalized grid (0, H/W-1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.

    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    # num_batch = tf.shape(theta)[0]

    # create 2D grid
    x = tf.linspace(0.0, tf.cast(width-1, 'float32'), width)
    y = tf.linspace(0.0, tf.cast(height-1, 'float32'), height)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float64')
    sampling_grid = tf.cast(sampling_grid, 'float64')

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids


def _hermite(A, B, C, D, t):
    """
    cubic Hermite spline interpolation given neighboring points
    Ref: https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    ┌─────────────────────────────────────────────┐
    │ -- samples given by four neighboring pixels │
    │ == destination to interpolate               │
    └─────────────────────────────────────────────┘
              B
     A        o  f(t)  C
     o        |    ║   o        D
     |        |    ║   |        o
     |--------|----║---|--------|   
     -1       0    t   1        2
    """

    a = A * (-0.5) + B * 1.5 + C * (-1.5) + D * 0.5
    b = A + B * (-2.5) + C * 2.0 + D * (-0.5)
    c = A * (-0.5) + C * 0.5
    d = B

    return a*t*t*t + b*t*t + c*t + d


def bicubic_sampler(img, x, y):
    """
    Performs bicubic sampling of the input images according to the
    coordinates provided by the sampling grid. Note that the samp-
    ling is done identically for each channel of the input.

    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.

    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """

    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    x2 = x0 + 2
    x_ = x0 - 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    y2 = y0 + 2
    y_ = y0 - 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    x2 = tf.clip_by_value(x2, zero, max_x)
    x_ = tf.clip_by_value(x_, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    y2 = tf.clip_by_value(y2, zero, max_y)
    y_ = tf.clip_by_value(y_, zero, max_y)
    x  = tf.clip_by_value(x, tf.cast(zero, 'float32'), tf.cast(max_x, 'float32'))
    y  = tf.clip_by_value(y, tf.cast(zero, 'float32'), tf.cast(max_y, 'float32'))

    # get pixel value at corner coords
    I__ = get_pixel_value(img, x_, y_)
    I0_ = get_pixel_value(img, x0, y_)
    I1_ = get_pixel_value(img, x1, y_)
    I2_ = get_pixel_value(img, x2, y_)
    I_0 = get_pixel_value(img, x_, y0)
    I00 = get_pixel_value(img, x0, y0)
    I10 = get_pixel_value(img, x1, y0)
    I20 = get_pixel_value(img, x2, y0)
    I_1 = get_pixel_value(img, x_, y1)
    I01 = get_pixel_value(img, x0, y1)
    I11 = get_pixel_value(img, x1, y1)
    I21 = get_pixel_value(img, x2, y1)
    I_2 = get_pixel_value(img, x_, y2)
    I02 = get_pixel_value(img, x0, y2)
    I12 = get_pixel_value(img, x1, y2)
    I22 = get_pixel_value(img, x2, y2)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    tx = x - x0
    ty = y - y0
    tx = tf.expand_dims(tx, axis=3)
    ty = tf.expand_dims(ty, axis=3)

    col0 = _hermite(I__, I0_, I1_, I2_, tx)
    col1 = _hermite(I_0, I00, I10, I20, tx)
    col2 = _hermite(I_1, I01, I11, I21, tx)
    col3 = _hermite(I_2, I02, I12, I22, tx)
    out  = _hermite(col0, col1, col2, col3, ty)

    return out  


class ResBlock(tf.keras.layers.Layer):
  def __init__(self, filters):
    super(ResBlock, self).__init__()
    self.filters = filters     
    self.conv1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
    self.bnorm1 = tf.keras.layers.BatchNormalization()
    self.act1   = tf.keras.layers.Activation('relu')  
    
    self.conv2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))
    self.bnorm2 = tf.keras.layers.BatchNormalization()
    self.act2   = tf.keras.layers.Activation('relu') 
    
    self.conv3 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02))   
    self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
    self.act3   = tf.keras.layers.Activation('relu')
  
  def call(self, x):
    inpt = x    
    c1  = self.conv1(x)
    bn1 = self.bnorm1(c1)
    bn1 = self.act1(bn1)   
    
    c2  = self.conv2(bn1)
    bn2 = self.bnorm2(c2)
    bn2 = self.act2(bn2) 
      
    c3  = self.conv3(inpt)
    a4  = bn2 + c3
    
    output = self.maxpool(a4)
    output = self.act3(output)   
    return output

   
class ResizeParamNet(tf.keras.Sequential):
  def __init__(self):
    super().__init__()   
    self.add(ResBlock(16))
    self.add(ResBlock(32))
    self.add(ResBlock(64))
    
    self.add(tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1,2,3])))
    self.add(tf.keras.layers.Lambda(lambda x: tf.reshape(x, [-1, 1])))
    self.add(tf.keras.layers.Activation('relu'))          

        
class ConvnextPreFilter(tf.keras.layers.Layer):
  def __init__(self, channels=3):
    super(ConvnextPreFilter, self).__init__()   
    depth = 3   
    self.convnextBlock = tf.keras.Sequential([
			     *[Block(
				  dim=channels,
				  drop_path=0.0,
				  layer_scale_init_value=1e-6,
				  name=f"convnext_block_{j}",
			       )
			       for j in range(depth)]
		          ],)      

  def call(self, x):
    inpt = x     
    x = self.convnextBlock(x)   
    output = inpt + x  
    return output   



class ConvnextPostFilter(tf.keras.layers.Layer):
  def __init__(self, channels=3):
    super(ConvnextPostFilter, self).__init__()   
    depth = 3   
    self.convnextBlock = tf.keras.Sequential([
			     *[Block(
				  dim=channels,
				  drop_path=0.0,
				  layer_scale_init_value=1e-6,
				  name=f"convnext_block_{j}",
			       )
			       for j in range(depth)]
		          ],)      

  def call(self, x):
    inpt = x     
    x = self.convnextBlock(x)   
    output = inpt + x  
    return output             
