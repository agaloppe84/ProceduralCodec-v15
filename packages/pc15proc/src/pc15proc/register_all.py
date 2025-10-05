from .registry import register
from .stripes import GEN as STRIPES
from .checkerboard import GEN as CHECKER
from .grad_linear import GEN as GRADIENT_LINEAR
from .grad_radial import GEN as GRADIENT_RADIAL
from .rings import GEN as RINGS
from .waves_dir import GEN as WAVES_DIR
from .waves_circ import GEN as WAVES_CIRC
from .plaid import GEN as PLAID
from .dots import GEN as DOTS
from .diamond_grid import GEN as DIAMOND_GRID
from .hex_grid import GEN as HEX_GRID
from .herringbone import GEN as HERRINGBONE
from .tri_tiling import GEN as TRI_TILING
from .moire import GEN as MOIRE
from .spiral import GEN as SPIRAL
from .value_noise import GEN as VALUE_NOISE
from .perlin import GEN as PERLIN
from .worley import GEN as WORLEY
from .marble import GEN as MARBLE
from .wood import GEN as WOOD
from .sunburst import GEN as SUNBURST
from .brick import GEN as BRICK
from .concentric_squares import GEN as CONCENTRIC_SQUARES

def register_all():
    for g in (STRIPES, CHECKER, GRADIENT_LINEAR, GRADIENT_RADIAL, RINGS,
              WAVES_DIR, WAVES_CIRC, PLAID, DOTS, DIAMOND_GRID, HEX_GRID,
              HERRINGBONE, TRI_TILING, MOIRE, SPIRAL, VALUE_NOISE, PERLIN,
              WORLEY, MARBLE, WOOD, SUNBURST, BRICK, CONCENTRIC_SQUARES):
        register(g)
