import torch

from .base_frontend import ScatteringBase2DMul
# from ...scattering2d.core.scattering2d import scattering2d
from ...scattering2dmul.core.scattering2dmul import scattering2dmul
from ...frontend.torch_frontend import ScatteringTorch

# class ScatteringTorch2D(ScatteringTorch, ScatteringBase2D):
class ScatteringTorch2DMul(ScatteringTorch, ScatteringBase2DMul):
    # def __init__(self, J, shape, L=8, max_order=2, pre_pad=False,
    # def __init__(self, J, shape, L=8, max_order=2, pre_pad=False,
    def __init__(self, J, shape, L=6, max_order=2, pre_pad=False,
            backend='torch', out_type='array'):
        ScatteringTorch.__init__(self)
        # ScatteringBase2D.__init__(**locals())
        # ScatteringBase2D._instantiate_backend(self, 'kymatio.scattering2d.backend.')
        # ScatteringBase2D.build(self)
        # ScatteringBase2D.create_filters(self)
        ScatteringBase2DMul.__init__(**locals())
        ScatteringBase2DMul._instantiate_backend(self, 'kymatio.scattering2d.backend.')
        ScatteringBase2DMul.build(self)
        ScatteringBase2DMul.create_filters(self)

        self.register_filters()

    def register_single_filter(self, v, n):
        current_filter = torch.from_numpy(v).unsqueeze(-1)
        self.register_buffer('tensor' + str(n), current_filter)
        return current_filter

    def register_filters(self):
        """ This function run the filterbank function that
            will create the filters as numpy array, and then, it
            saves those arrays as module's buffers."""
        # Create the filters

        n = 0

        for c, phi in self.phi.items():
            if not isinstance(c, int):
                continue

            self.phi[c] = self.register_single_filter(phi, n)
            n = n + 1

        for j in range(len(self.psi)):
            for k, v in self.psi[j].items():
                if not isinstance(k, int):
                    continue

                self.psi[j][k] = self.register_single_filter(v, n)
                n = n + 1

    def load_single_filter(self, n, buffer_dict):
        return buffer_dict['tensor' + str(n)]

    def load_filters(self):
        """ This function loads filters from the module's buffers """
        # each time scattering is run, one needs to make sure self.psi and self.phi point to
        # the correct buffers
        buffer_dict = dict(self.named_buffers())

        n = 0

        phis = self.phi
        for c, phi in phis.items():
            if not isinstance(c, int):
                continue

            phis[c] = self.load_single_filter(n, buffer_dict)
            n = n + 1

        psis = self.psi
        for j in range(len(psis)):
            for k, v in psis[j].items():
                if not isinstance(k, int):
                    continue

                psis[j][k] = self.load_single_filter(n, buffer_dict)
                n = n + 1

        return phis, psis

    def scattering(self, input):
        if not torch.is_tensor(input):
            raise TypeError('The input should be a PyTorch Tensor.')

        if len(input.shape) < 2:
            raise RuntimeError('Input tensor must have at least two dimensions.')

        if not input.is_contiguous():
            raise RuntimeError('Tensor must be contiguous.')

        if (input.shape[-1] != self.N or input.shape[-2] != self.M) and not self.pre_pad:
            raise RuntimeError('Tensor must be of spatial size (%i,%i).' % (self.M, self.N))

        if (input.shape[-1] != self.N_padded or input.shape[-2] != self.M_padded) and self.pre_pad:
            raise RuntimeError('Padded tensor must be of spatial size (%i,%i).' % (self.M_padded, self.N_padded))

        if not self.out_type in ('array', 'list'):
            raise RuntimeError("The out_type must be one of 'array' or 'list'.")

        phi, psi = self.load_filters()

        batch_shape = input.shape[:-2]
        signal_shape = input.shape[-2:]

        input = input.reshape((-1,) + signal_shape)

        # S = scattering2d(input, self.pad, self.unpad, self.backend, self.J,
        # S = scattering2dmul(input, self.pad, self.unpad, self.backend, self.J,
        #                    self.L, phi, psi, self.max_order, self.out_type)
        S1, S2, S3 = scattering2dmul(input, self.pad, self.unpad, self.backend, self.J,
                           self.L, phi, psi, self.max_order, self.out_type)

        if self.out_type == 'array':
            scattering_shape_1 = S1.shape[-3:] # [81,8,8]
            S1 = S1.reshape(batch_shape + scattering_shape_1)

            scattering_shape_2 = S2.shape[-3:] # [81,8,8]
            S2 = S2.reshape(batch_shape + scattering_shape_2)

            scattering_shape_3 = S3.shape[-3:] # [81,8,8]
            S3 = S3.reshape(batch_shape + scattering_shape_3)
        else:
            scattering_shape_1 = S1[0]['coef'].shape[-2:]
            scattering_shape_2 = S2[0]['coef'].shape[-2:]
            scattering_shape_3 = S3[0]['coef'].shape[-2:]
            # new_shape = batch_shape + scattering_shape
            for x in S1:
                x['coef'] = x['coef'].reshape(batch_shape + scattering_shape_1)
            for x in S2:
                x['coef'] = x['coef'].reshape(batch_shape + scattering_shape_2)
            for x in S3:
                x['coef'] = x['coef'].reshape(batch_shape + scattering_shape_3)

        return S1, S2, S3

# ScatteringTorch2D._document()
ScatteringTorch2DMul._document()


# __all__ = ['ScatteringTorch2D']
__all__ = ['ScatteringTorch2DMul']
