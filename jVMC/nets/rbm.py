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

    Dense_with : int = 1

    @nn.compact
    def __call__(self, x): # input has dim (L**2,) or (L,) defdending non init in Filter its 2d or 1d

        # initFunction = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform", dtype=global_defs.tReal)

        # initFunction = jVMC.nets.initializers.cplx_variance_scaling

        initFunction_dense = partial(jVMC.nets.initializers.cplx_variance_scaling_dense, dense_width=self.Dense_with)

        # initFunction_dense=jVMC.nets.initializers.cplx_dense_ones

        initFunction = jVMC.nets.initializers.cplx_init2

        # initFunction_dense = initFunction 

        bias = [self.bias] * len(self.channels)

        activationFunctions = [f for f in self.actFun]
        for l in range(len(activationFunctions), len(self.channels)):
            activationFunctions.append(self.actFun[-1])

        init_args = init_fn_args(dtype=global_defs.tCpx, kernel_init=initFunction)
        init_args_dense = init_fn_args(dtype=global_defs.tCpx, kernel_init=initFunction_dense)

        # List of axes that will be summed for symmetrization
        # reduceDims = tuple([-i - 1 for i in range(len(self.strides) + 2)])

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
                            use_bias=b, **init_args)(x)) #shape ist [1,L,L,ch] danach
        
        #eine abschließende Dense layer wird über letze dimesin gemach, batchdimension verscheindet eh, und nach dense laser haben wir ein lx x ly bild. theorteishc könnte man vorher auch alle bilder zusammen summeiren und danach man ende ins dense
        # batch_size = x.shape[0]
        # x = x.reshape(batch_size, -1)
        # x = jnp.ravel(x)
        

        # jax.debug.print("CNN_sumx={x}",x=jnp.sum(x))
        # # vorher die bilder summieren
        # x = jnp.sum(x, axis=( 1, 2))


        # vorher die channels summieren (udn batch)
        x = jnp.sum(x, axis=(0,3))
        x = jnp.ravel(x)


    
        x = nn.Dense(self.Dense_with,**init_args_dense)(x) # w*X + w_2*X ,X hier dei bilder [1,L,L]

        

        return  jnp.sum(x) # just to get the right shaoe and remove batch dim
    
######################################

class ResNetV2(nn.Module):
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
    F: Sequence[int] = (3,3)
    channels: Sequence[int] = (10,10)
    strides: Sequence[int] = (1,1)
    actFun: Sequence[callable] = (act_funs.poly6,)
    bias: bool = False
    firstLayerBias: bool = False
    Lx : int = None
    Ly : int = None

    Dense_with : int = 1 
    
    nblocks : int = 1

    @nn.compact
    def __call__(self, x): # input has dim (L**2,) or (L,) defdending non init in Filter its 2d or 1d

        def pad_periodic_x_zero_y(x, kernel_size):
            
            # Warnung oder Anpassung bei gerader Kernelgröße
            if kernel_size % 2 == 0:
                raise ValueError(f"Warnung: Kernelgröße {kernel_size} ist gerade. Symmetrisches Padding führt zu vergrößertem Output. Residuals passen nicht mehr")
               

            pad = (kernel_size - 1)

            # In offener Richtung (y-Achse) unsymmetrisch paddden
            pad_y = int(pad / 2)
            # print(np.dtype(pad_y))
            x = jnp.pad(x, ((0, 0), (pad_y, pad_y), (0, 0), (0, 0)), mode='constant', constant_values=0)  # y-Achse

            # In periodischer Richtung (x-Achse) einfach rechts paddden
            x = jnp.pad(x, ((0, 0), (0, 0), (0, pad), (0, 0)), mode='wrap')  # x-Achse
            return x


        # initFunction = jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="uniform", dtype=global_defs.tReal)
        # initFunction = jVMC.nets.initializers.cplx_variance_scaling

        initFunction_dense = partial(jVMC.nets.initializers.cplx_variance_scaling_dense, dense_width=self.Dense_with)
        initFunction = jVMC.nets.initializers.cplx_init2

        init_args = init_fn_args(dtype=global_defs.tCpx, kernel_init=initFunction)
        init_args_dense = init_fn_args(dtype=global_defs.tCpx, kernel_init=initFunction_dense)


        # für jeden block die architectur  #########

        bias = [self.bias] * len(self.channels)

        activationFunctions = [f for f in self.actFun]
        for l in range(len(activationFunctions), len(self.channels)):
            activationFunctions.append(self.actFun[-1])

        # für 2D fall
        x = jnp.reshape(x, (self.Lx,self.Ly))

        # # Add feature dimension
        x = jnp.expand_dims(jnp.expand_dims(2 * x - 1, axis=0), axis=-1)

        # Berechnung 
        for nblock in range(self.nblocks):

            residual = x

            # Berechnung in jedem block
            for c, f, b in zip(self.channels, activationFunctions, bias):
                
                # aktiverungsfunktion für den block davor noch, beim letzten dann keine, erst nach dense layer wieder
                if nblock > 0:
                    x = f(x)

                x = pad_periodic_x_zero_y(x, kernel_size=self.F[nblock])

                x = nn.Conv(features=c, kernel_size=(self.F[nblock], self.F[nblock]),
                                strides=self.strides,
                                use_bias=b, padding="VALID", **init_args)(x) #shape ist [1,L,L,ch] danach
            
            x += residual

        # vorher die channels summieren (udn batch)
        x = jnp.sum(x, axis=(0,3))
        x = jnp.ravel(x)
    
        x = nn.Dense(self.Dense_with,**init_args_dense)(x) # w*X + w_2*X ,X hier dei bilder [1,L,L]

        # eine letzte activerungsfunktion
        x = activationFunctions[-1](x)

        

        return  jnp.sum(x) # just to get the right shaoe and remove batch dim



#######################################





class ResNet(nn.Module):
    F: Sequence[int] = (3,)
    channels: Sequence[int] = (16,)
    strides: Sequence[int] = (1,)
    bias: bool = True
    Lx: int = 4
    Ly: int = 4
    Dense_with: int = 1



    @nn.compact
    def __call__(self, x):
        def pad_periodic_x_zero_y(x, kernel_size):
            
            # Warnung oder Anpassung bei gerader Kernelgröße
            if kernel_size % 2 == 0:
                raise ValueError(f"Warnung: Kernelgröße {kernel_size} ist gerade. Symmetrisches Padding führt zu vergrößertem Output. Residuals passen nicht mehr")
               

            pad = (kernel_size - 1)

            # In offener Richtung (y-Achse) unsymmetrisch paddden
            pad_y = int(pad / 2)
            # print(np.dtype(pad_y))
            x = jnp.pad(x, ((0, 0), (pad_y, pad_y), (0, 0), (0, 0)), mode='constant', constant_values=0)  # y-Achse

            # In periodischer Richtung (x-Achse) einfach rechts paddden
            x = jnp.pad(x, ((0, 0), (0, 0), (0, pad), (0, 0)), mode='wrap')  # x-Achse
            return x
        
     

        # gewicht initialisierung dür die layers
        # CNN
        initFunction = jax.nn.initializers.he_normal(dtype=global_defs.tReal)
        # initFunction = jVMC.nets.initializers.real_init2

        #dense
        initFunction_dense = jax.nn.initializers.normal(stddev=1.0, dtype=global_defs.tReal)
        # initFunction_dense = partial(jVMC.nets.initializers.cplx_variance_scaling_dense, dense_width=self.Dense_with)
        
        nsites = x.size

        if x.ndim == 1:
            x = x.reshape(self.Lx, self.Ly)
        elif x.ndim == 2 and x.shape[0] == 1:
            x = jnp.squeeze(x, axis=0)
            x = x.reshape(self.Lx, self.Ly)
        x = jnp.expand_dims(jnp.expand_dims(2 * x - 1, axis=0), axis=-1)
        x = x.astype(global_defs.tReal)

        x_sym = -1*x


        for nblock in range(len(self.channels)):

            residual = x
            # print(x)
            # print("residual-shape",residual.shape)

            # also run teh flipped config trough the net to make it symetrick under spinflip
            residual_sym = x_sym
            

            norm_factor = jnp.sqrt(global_defs.tReal(nblock + 1))
            if nblock == 0:
                norm_factor *= jnp.sqrt(global_defs.tReal(2))
                x /= norm_factor

                x_sym /= norm_factor

                #x = act_funs.poly5(x) # erset soll symetrisch sein, weil wir -1 und 1 inpunts gleihcbehandeln wollen

            if nblock > 0:
    
                x = nn.gelu(x)
                # x = act_funs.poly6(x)

                x_sym = nn.gelu(x_sym)


            # schicke beide x und spinfip duch netz-----------------------------------
            x_both = jnp.concatenate([x, x_sym], axis=0)  # batch verdoppeln

            x_both = pad_periodic_x_zero_y(x_both, kernel_size=self.F[0])

            # x = pad_periodic_x_zero_y(x, kernel_size=self.F[0])

            # print("x-shape",x.shape)
            x_both = nn.Conv(
                features=self.channels[nblock],
                kernel_size=tuple(self.F),
                padding="VALID",
                strides=(self.strides[0], self.strides[0]),
                use_bias=self.bias,
                param_dtype=global_defs.tReal,
                dtype=global_defs.tReal,
                kernel_init=initFunction,
                bias_init=jax.nn.initializers.zeros,
            )(x_both)
           
            x_both = nn.gelu(x_both) 
            # x = act_funs.poly6(x)

            # x = pad_periodic_x_zero_y(x, kernel_size=self.F[0])
            x_both = pad_periodic_x_zero_y(x_both, kernel_size=self.F[0])

            x_both = nn.Conv(
                features=self.channels[nblock],
                kernel_size=tuple(self.F),
                padding="VALID",
                strides=(self.strides[0], self.strides[0]),
                use_bias=self.bias and (nblock != len(self.channels) - 1),
                param_dtype=global_defs.tReal,
                dtype=global_defs.tReal,
                kernel_init=initFunction,
                bias_init=jax.nn.initializers.zeros,
            )(x_both)

            # Split wieder in original und spinflipped
            x, x_sym = jnp.split(x_both, 2, axis=0)

            x += residual
            x_sym += residual_sym

        x_real = x[..., :(x.shape[-1] // 2)]
        x_imag = x[..., (x.shape[-1] // 2):]

        x_real_sym = x_sym[..., :(x_sym.shape[-1] // 2)]
        x_imag_sym = x_sym[..., (x_sym.shape[-1] // 2):]

        # Summe über Batch- und Kanalachsen
        x_real = jnp.sum(x_real, axis=(0, 3)).ravel()
        x_imag = jnp.sum(x_imag, axis=(0, 3)).ravel()

        x_real_sym = jnp.sum(x_real_sym, axis=(0, 3)).ravel()
        x_imag_sym = jnp.sum(x_imag_sym, axis=(0, 3)).ravel()

        # x_real und x_imag haben shape (features,)
        x_stacked = jnp.stack([x_real, x_imag, x_real_sym, x_imag_sym], axis=0)  # shape (2, features) 

        # missbrauchen der batchdim für real und komplex. jvmc erlaubt nur dtype= konst, und be CNN = global_defs.tReal, also auch bei Dense layer und nicht einfach dtype=global_defs.tCpx möglich 
        x_stacked = nn.Dense(
            self.Dense_with,
            use_bias=False,
            param_dtype=global_defs.tReal,
            dtype=global_defs.tReal,
            kernel_init = initFunction_dense,
        )(x_stacked)


        x_real_out = x_stacked[0]
        x_imag_out = x_stacked[1]

        x_real_out_sym =  x_stacked[2]
        x_imag_out_sym =  x_stacked[3]

        x_complex_out = jax.lax.complex(x_real_out, x_imag_out)
        x_complex_out_sym = jax.lax.complex(x_real_out_sym, x_imag_out_sym)

        # print("x_complex_out-shape",x_complex_out.shape)

        x_complex_out = jax.scipy.special.logsumexp(x_complex_out) - jnp.log(self.channels[nblock]*nsites)
        x_complex_out_sym = jax.scipy.special.logsumexp(x_complex_out_sym) - jnp.log(self.channels[nblock]*nsites)


        #symterisieren
        x_complex_out_final = 1/2*(x_complex_out_sym + x_complex_out)


        return jnp.sum(x_complex_out_final)