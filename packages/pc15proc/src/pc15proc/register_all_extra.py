from .registry import register
# Previous set expected to be already present; here we only register the extra 20.
from .gabor import GEN as GABOR
from .gaussian_spots import GEN as GAUSSIAN_SPOTS
from .voronoi_edges import GEN as VORONOI_EDGES
from .voronoi_cells import GEN as VORONOI_CELLS
from .fbm_ridged import GEN as FBM_RIDGED
from .turbulence import GEN as TURBULENCE
from .gabor_bank import GEN as GABOR_BANK
from .weave import GEN as WEAVE
from .lattice import GEN as LATTICE
from .polar_checker import GEN as POLAR_CHECKER
from .starfield import GEN as STARFIELD
from .ripple import GEN as RIPPLE
from .waves_bend import GEN as WAVES_BEND
from .swirl import GEN as SWIRL
from .checker_distorted import GEN as CHECKER_DISTORTED
from .stripes_multiscale import GEN as STRIPES_MULTISCALE
from .isobands import GEN as ISOBANDS
from .pebbles import GEN as PEBBLES
from .brick_noise import GEN as BRICK_NOISE
from .kaleidoscope import GEN as KALEIDOSCOPE
from .angle_gradient import GEN as ANGLE_GRADIENT
from .spiral_square import GEN as SPIRAL_SQUARE
from .gabor_noise import GEN as GABOR_NOISE

def register_all_extra():
    for g in (GABOR, GAUSSIAN_SPOTS, VORONOI_EDGES, VORONOI_CELLS, FBM_RIDGED,
              TURBULENCE, GABOR_BANK, WEAVE, LATTICE, POLAR_CHECKER, STARFIELD,
              RIPPLE, WAVES_BEND, SWIRL, CHECKER_DISTORTED, STRIPES_MULTISCALE,
              ISOBANDS, PEBBLES, BRICK_NOISE, KALEIDOSCOPE, ANGLE_GRADIENT,
              SPIRAL_SQUARE, GABOR_NOISE):
        register(g)
