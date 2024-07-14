"""Contains the factories for selecting different models

Author: Benjamin Therien

Functions: 
    baseModelFactory -- Factory for the creation of the first part of a hybrid model
    topModelFactory -- Factory for the creation of seconds part of a hybrid model


Author: Dawn

Functions:
    baseMultiModelFactory -- Factory for the multi-level scattering operator of a hybrid model
    topSepaModelFactory -- Factory for the separate head attention operator of a hybrid model
"""

from .sn_top_models import sn_CNN, sn_MLP, sn_LinearLayer, sn_Resnet50, sn_ViT, sn_MulViT
from .sn_mid_models import mid_CNN, mid_MLP, mid_LinearLayer, mid_Resnet50, mid_ViT, mid_Atten

from .sn_base_multibatch_models import sn_Identity, sn_ScatteringBase, sn_MultiScatteringBase

class InvalidArchitectureError(Exception):
    """Error thrown when an invalid architecture name is passed"""
    pass

def baseModelFactory(architecture, J, N, M, second_order, initialization, seed, 
                     learnable=True, lr_orientation=0.1, lr_scattering=0.1,
                     filter_video=False, parameterization='canonical'):
    """Factory for the creation of the first layer of a hybrid model
    
        parameters: 
            J -- scale of scattering
            N -- height of the input image
            M -- width of the input image
            initilization -- the type of init
            lr_orientation -- learning rate for the orientation of the scattering parameters
            lr_scattering -- learning rate for scattering parameters other than orientation                 
    """

    if architecture.lower() == 'identity':
        return sn_Identity()

    elif architecture.lower() == 'scattering':
        return sn_ScatteringBase( #create learnable of non-learnable scattering
            J=J,
            N=N,
            M=M,
            second_order=second_order,
            initialization=initialization,
            seed=seed,
            learnable=learnable,
            lr_orientation=lr_orientation,
            lr_scattering=lr_scattering,
            filter_video=filter_video,
            parameterization= parameterization
        )

    elif architecture.lower() == 'scatteringmul':
        return sn_MultiScatteringBase( #create learnable of non-learnable scattering
            J=J,
            N=N,
            M=M,
            second_order=second_order,
            initialization=initialization,
            seed=seed,
            learnable=learnable,
            lr_orientation=lr_orientation,
            lr_scattering=lr_scattering,
            filter_video=filter_video,
            parameterization= parameterization
        )

    else:
        print("In modelFactory() incorrect module name for architecture={}".format(architecture))
        raise InvalidArchitectureError()


def baseMultiModelFactory(architecture, J, N, M, second_order, initialization, seed,
                     learnable=True, lr_orientation=0.1, lr_scattering=0.1,
                     filter_video=False, parameterization='canonical'):
    """Factory for the multi-level scattering operator of a hybrid model

        parameters:
            J -- scale of scattering
            N -- height of the input image
            M -- width of the input image
            lr_orientation -- learning rate for the orientation of the scattering parameters
            lr_scattering -- learning rate for scattering parameters other than orientation
    """
    if architecture.lower() == 'identity':
        return sn_Identity()

    elif architecture.lower() == 'scattering':
        return sn_MultiScatteringBase(  # create learnable of non-learnable scattering
            J=J,
            N=N,
            M=M,
            second_order=second_order,
            initialization=initialization,
            seed=seed,
            learnable=learnable,
            lr_orientation=lr_orientation,
            lr_scattering=lr_scattering,
            filter_video=filter_video,
            parameterization=parameterization
        )

    else:
        print("In modelFactory() incorrect module name for architecture={}".format(architecture))
        raise InvalidArchitectureError()


def topModelFactory(base, architecture, num_classes, width=8, average=False):
    """Factory for the creation of seconds part of a hybrid model
    
    parameters:
        base         -- (Pytorch nn.Module) the first part of a hybrid model
        architecture -- the name of the top model to select
        num_classes  -- number of classes in dataset
        width        -- the width of the model
    """

    if architecture.lower() == 'cnn':
        return sn_CNN(
            base.n_coefficients, k=width, num_classes=num_classes, standard=False
        )

    elif architecture.lower() == 'mlp':
        return sn_MLP(
            num_classes=num_classes, n_coefficients=base.n_coefficients, 
            M_coefficient=base.M_coefficient, N_coefficient=base.N_coefficient
        )

    elif architecture.lower() == 'linear_layer':
        return sn_LinearLayer(
            num_classes=num_classes, n_coefficients=base.n_coefficients, 
            M_coefficient=base.M_coefficient, N_coefficient=base.N_coefficient, 
        )

    elif architecture.lower() == 'resnet50':
        return sn_Resnet50(num_classes=num_classes)

    elif architecture.lower() == 'vit':

        return sn_ViT(base.n_coefficients, num_classes=num_classes, standard=False)
        # return sn_ViT(n_coefficients=base.n_coefficients, num_heads=num_heads, num_layers=num_layers,
        #               num_classes=num_classes, standard=False)

    elif architecture.lower() == 'vitonly':
        return sn_ViT(base.n_coefficients, num_classes=num_classes, standard=True)

    elif architecture.lower() == 'mulvit':
        # return sn_ViT(n_coefficients=base.n_coefficients, num_classes=num_classes, standard=True)
        return sn_MulViT(base.n_coefficients, num_classes=num_classes, standard=False)

    else:
        print("In modelFactory() incorrect module name for architecture={}".format(architecture))
        raise InvalidArchitectureError()


def midModelFactory(base, architecture, num_classes, width=8, average=False, num_layers=3):

    if architecture.lower() == 'cnn':
        return mid_CNN(
            base.n_coefficients, k=width, num_classes=num_classes, standard=False
        )

    elif architecture.lower() == 'mlp':
        return mid_MLP(
            num_classes=num_classes, n_coefficients=base.n_coefficients,
            M_coefficient=base.M_coefficient, N_coefficient=base.N_coefficient
        )

    elif architecture.lower() == 'linear_layer':
        return mid_LinearLayer(
            num_classes=num_classes, n_coefficients=base.n_coefficients,
            M_coefficient=base.M_coefficient, N_coefficient=base.N_coefficient,
        )

    # elif architecture.lower() == 'resnet50':
    #     return sn_Resnet50(num_classes=num_classes)
    elif architecture.lower() == 'atten':
        # return sn_ViT(n_coefficients=base.n_coefficients, num_classes=num_classes, standard=True)
        return mid_Atten(base.n_coefficients, num_layers=num_layers, standard=False)


    else:
        print("In modelFactory() incorrect module name for architecture={}".format(architecture))
        raise InvalidArchitectureError()

