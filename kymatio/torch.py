from .scattering1d.frontend.torch_frontend import ScatteringTorch1D as Scattering1D
from .scattering2d.frontend.torch_frontend import ScatteringTorch2D as Scattering2D
from .scattering2dmul.frontend.torch_frontend import ScatteringTorch2DMul as Scattering2DMul
from .scattering3d.frontend.torch_frontend \
        import HarmonicScatteringTorch3D as HarmonicScattering3D

Scattering1D.__module__ = 'kymatio.torch'
Scattering1D.__name__ = 'Scattering1D'

Scattering2D.__module__ = 'kymatio.torch'
Scattering2D.__name__ = 'Scattering2D'

Scattering2DMul.__module__ = 'kymatio.torch'
Scattering2DMul.__name__ = 'Scattering2DMul'

HarmonicScattering3D.__module__ = 'kymatio.torch'
HarmonicScattering3D.__name__ = 'HarmonicScattering3D'

#__all__ = ['Scattering1D', 'Scattering2D', 'HarmonicScattering3D']
__all__ = ['Scattering2DMul', 'Scattering1D', 'Scattering2D', 'HarmonicScattering3D']
