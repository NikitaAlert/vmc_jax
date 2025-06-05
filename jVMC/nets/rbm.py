import jax
jax.config.update("jax_enable_x64", True)
import flax
#from flax import nn
import flax.linen as nn
import jax.numpy as jnp

import jVMC.global_defs as global_defs
import jVMC.nets.activation_functions as act_funs
from jVMC.nets.initializers import init_fn_args

from functools import partial

import jVMC.nets.initializers


# additional for Wladis CpxRBMCNN-achitecture
from typing import List, Sequence
import jVMC.global_defs as global_defs
import jVMC.nets.initializers
from jVMC.util.symmetries import LatticeSymmetry


class CpxRBM(nn.Module):
    """Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias,
                         **init_fn_args(kernel_init=jVMC.nets.initializers.cplx_init,
                                        bias_init=jax.nn.initializers.zeros,
                                        dtype=global_defs.tCpx)
                         )

        return jnp.sum(act_funs.log_cosh(layer(2 * s.ravel() - 1)))

# ** end class CpxRBM


class CpxRBM_Nospinflip(nn.Module):
    """Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias,
                         **init_fn_args(kernel_init=jVMC.nets.initializers.cplx_init,
                                        bias_init=jax.nn.initializers.zeros,
                                        dtype=global_defs.tCpx)
                         )

        return jnp.sum(act_funs.log_cosh(layer(s.ravel())))


class RBM(nn.Module):
    """Restricted Boltzmann machine with real parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(self.numHidden, use_bias=self.bias,
                         **init_fn_args(kernel_init=jax.nn.initializers.lecun_normal(dtype=global_defs.tReal),
                                        bias_init=jax.nn.initializers.zeros,
                                        dtype=global_defs.tReal)
                        )

        return jnp.sum(jnp.log(jnp.cosh(layer(2 * s - 1))))

# ** end class RBM

class CpxRBMCNN(nn.Module):
    """Convolutional neural network with complex parameters.

    Initialization arguments:
        * ``F``: Filter diameter
        * ``channels``: Number of channels
        * ``strides``: Number of pixels the filter shifts over
        * ``actFun``: Non-linear activation function
        * ``bias``: Whether to use biases
        * ``firstLayerBias``: Whether to use biases in the first layer
        * ``periodicBoundary``: Whether to use periodic boundary conditions

    """
    F: Sequence[int] = (8,)
    channels: Sequence[int] = (10,)
    strides: Sequence[int] = (1,)
    actFun: Sequence[callable] = (act_funs.log_cosh,)
    bias: bool = False
    firstLayerBias: bool = False
    periodicBoundary: Sequence[bool] = (True,False) # Zylinder
    Lx : int = None
    Ly : int = None

    @nn.compact
    def __call__(self, x): # input has dim (L**2,) or (L,) defdending non init in Filter its 2d or 1d

        # initFunction = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform", dtype=global_defs.tReal)
        # initFunction = jVMC.nets.initializers.cplx_variance_scaling
        initFunction = jVMC.nets.initializers.cplx_init

        bias = [self.bias] * len(self.channels)

        activationFunctions = [f for f in self.actFun]
        for l in range(len(activationFunctions), len(self.channels)):
            activationFunctions.append(self.actFun[-1])

        init_args = init_fn_args(dtype=global_defs.tCpx, kernel_init=initFunction)

        # List of axes that will be summed for symmetrization
        reduceDims = tuple([-i - 1 for i in range(len(self.strides) + 2)])

        # für 2D fall
        if not self.Lx==None:
            x = jnp.reshape(x, (self.Lx,self.Ly))

        # # Add feature dimension
        x = jnp.expand_dims(jnp.expand_dims(2 * x - 1, axis=0), axis=-1)


        # Set up padding for periodic boundary conditions
        # Padding size must be 1 - filter diameter
        # 1D
        if self.Lx==None:
            if self.periodicBoundary[0]:
                x = jnp.pad(x, [(0, 0), (0, self.F[0] - 1), (0, 0)], 'wrap')
                # print(x[0,:,:,0])
            else:
                x = jnp.pad(x, [(0, 0), (self.F[0] - 1, self.F[0] - 1), (0, 0)], 'constant', constant_values=0)
                # print(x[0,:,:,0])
        #2D
        if not self.Lx==None:


            # Padding in x-Richtung (Achse 2 !! nicht achse 1) # Also, wenn du Periodizität in X (horizontal) möchtest, dann bedeutet das: Die zweite Achse (Spalten) sollte periodisch behandelt werden
            if self.periodicBoundary[0]:
                x = jnp.pad(x, [(0, 0), (0, 0), (0, self.F[0] - 1), (0, 0)], mode="wrap")
            else:
                x = jnp.pad(x, [(0, 0), (0, 0), ((self.F[0] - 1, self.F[0] - 1 )), (0, 0)], mode="constant", constant_values=0) # bei nicht periodischen muss links und rechtes beiden seiten mit 0llen einfach
                # asymmetrisch korrekt

            # Padding in y-Richtung (Achse 1, F[1]!)  
            if self.periodicBoundary[1]:
                x = jnp.pad(x, [(0, 0), (0, self.F[1] - 1), (0, 0), (0, 0)], mode="wrap")
            else:
                x = jnp.pad(x, [(0, 0), (self.F[1] - 1, self.F[1] - 1 ), (0, 0), (0, 0)], mode="constant", constant_values=0)  



        # Berechnung 
        for c, f, b in zip(self.channels, activationFunctions, bias):

            x = f(nn.Conv(features=c, kernel_size=tuple(self.F),
                            strides=self.strides,
                            use_bias=b, **init_args)(x))

        return jnp.sum(x)


######## Erweiterung

class CpxCNNDense(nn.Module):
    """Convolutional neural network with complex parameters.

    Initialization arguments:
        * ``F``: Filter diameter
        * ``channels``: Number of channels
        * ``strides``: Number of pixels the filter shifts over
        * ``actFun``: Non-linear activation function
        * ``bias``: Whether to use biases
        * ``firstLayerBias``: Whether to use biases in the first layer
        * ``periodicBoundary``: Whether to use periodic boundary conditions

    """
    F: Sequence[int] = (8,)
    channels: Sequence[int] = (10,)
    strides: Sequence[int] = (1,)
    actFun: Sequence[callable] = (act_funs.log_cosh,)
    bias: bool = False
    firstLayerBias: bool = False
    periodicBoundary: Sequence[bool] = (True,False) # Zylinder
    Lx : int = None
    Ly : int = None

    @nn.compact
    def __call__(self, x): # input has dim (L**2,) or (L,) defdending non init in Filter its 2d or 1d

        # initFunction = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform", dtype=global_defs.tReal)
        # initFunction = jVMC.nets.initializers.cplx_variance_scaling
        initFunction = jVMC.nets.initializers.cplx_init

        bias = [self.bias] * len(self.channels)

        activationFunctions = [f for f in self.actFun]
        for l in range(len(activationFunctions), len(self.channels)):
            activationFunctions.append(self.actFun[-1])

        init_args = init_fn_args(dtype=global_defs.tCpx, kernel_init=initFunction)

        # List of axes that will be summed for symmetrization
        reduceDims = tuple([-i - 1 for i in range(len(self.strides) + 2)])

        # für 2D fall
        if not self.Lx==None:
            x = jnp.reshape(x, (self.Lx,self.Ly))

        # # Add feature dimension
        x = jnp.expand_dims(jnp.expand_dims(2 * x - 1, axis=0), axis=-1)


        # Set up padding for periodic boundary conditions
        # Padding size must be 1 - filter diameter
        # 1D
        if self.Lx==None:
            if self.periodicBoundary[0]:
                x = jnp.pad(x, [(0, 0), (0, self.F[0] - 1), (0, 0)], 'wrap')
                # print(x[0,:,:,0])
            else:
                x = jnp.pad(x, [(0, 0), (self.F[0] - 1, self.F[0] - 1), (0, 0)], 'constant', constant_values=0)
                # print(x[0,:,:,0])
        #2D
        if not self.Lx==None:


            # Padding in x-Richtung (Achse 2 !! nicht achse 1) # Also, wenn du Periodizität in X (horizontal) möchtest, dann bedeutet das: Die zweite Achse (Spalten) sollte periodisch behandelt werden
            if self.periodicBoundary[0]:
                x = jnp.pad(x, [(0, 0), (0, 0), (0, self.F[0] - 1), (0, 0)], mode="wrap")
            else:
                x = jnp.pad(x, [(0, 0), (0, 0), ((self.F[0] - 1, self.F[0] - 1 )), (0, 0)], mode="constant", constant_values=0) # bei nicht periodischen muss links und rechtes beiden seiten mit 0llen einfach
                # asymmetrisch korrekt

            # Padding in y-Richtung (Achse 1, F[1]!)  
            if self.periodicBoundary[1]:
                x = jnp.pad(x, [(0, 0), (0, self.F[1] - 1), (0, 0), (0, 0)], mode="wrap")
            else:
                x = jnp.pad(x, [(0, 0), (self.F[1] - 1, self.F[1] - 1 ), (0, 0), (0, 0)], mode="constant", constant_values=0)  



        # Berechnung 
        for c, f, b in zip(self.channels, activationFunctions, bias):

            x = f(nn.Conv(features=c, kernel_size=tuple(self.F),
                            strides=self.strides,
                            use_bias=b, **init_args)(x))
        
        #eine abschließende Dense layer
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = activationFunctions[-1](nn.Dense(self.channels[-1])(x))

        return jnp.sum(x)