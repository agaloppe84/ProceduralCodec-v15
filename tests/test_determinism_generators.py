import time
import logging
import torch

from pc15proc.register_all import register_all
from pc15proc.registry import list_generators, get
from pc15proc.params import ParamCodec


# Logger dédié à ce test
log = logging.getLogger("pc15.tests.determinism")
if not logging.getLogger().handlers:
    # Fallback local ; sous pytest, la config de log peut être override.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def dev():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mid_params(info):
    out = {}
    for p in info.param_specs:
        if p.type in ("float", "int") and p.range:
            lo, hi = p.range
            x = (float(lo) + float(hi)) / 2.0
            out[p.name] = int(round(x)) if p.type == "int" else x
        elif p.type == "enum":
            out[p.name] = p.enum[0]
        elif p.type == "bool":
            out[p.name] = False
        else:
            out[p.name] = 0.0
    return out


def test_determinism_subset():
    register_all()
    device = dev()
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    H, W = 64, 64
    seeds = torch.tensor([42], device=device, dtype=torch.int64)

    gens = list_generators()
    log.info("Starting determinism sweep: device=%s, dtype=%s, HxW=%dx%d, generators=%d",
             device, dtype, H, W, len(gens))

    for gi in gens:
        try:
            g = get(gi.name)
            pc = ParamCodec(g.info)
            params_dict = mid_params(g.info)

            # Log compact des params (évite d'inonder les logs avec des gros dicts)
            preview = ", ".join(f"{k}={v}" for k, v in list(params_dict.items())[:6])
            if len(params_dict) > 6:
                preview += ", ..."

            P = pc.to_tensor(params_dict, device=device, dtype=dtype).unsqueeze(0)

            t0 = time.perf_counter()
            y1 = g.render((H, W), P, seeds, device=device, dtype=dtype)
            t1 = time.perf_counter()
            y2 = g.render((H, W), P, seeds, device=device, dtype=dtype)
            t2 = time.perf_counter()

            # Stats pour debug
            y1c = y1.detach().to(torch.float32).cpu()
            mn = float(y1c.min())
            mx = float(y1c.max())
            mean = float(y1c.mean())
            checksum = float(y1c.sum())  # checksum simple mais pratique

            log.info(
                "GEN=%-18s | params={%s} | t1=%.3f ms t2=%.3f ms | shape=%s | min=%.4f max=%.4f mean=%.4f sum=%.4f",
                gi.name, preview, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0,
                tuple(y1.shape), mn, mx, mean, checksum
            )

            # Assertions utiles pour attraper les régressions rapidement
            assert y1.shape == (1, 1, H, W), f"Bad shape for {gi.name}: {y1.shape}"
            assert torch.isfinite(y1).all(), f"NaN/Inf in output: {gi.name}"
            assert (y1 >= -1).all() and (y1 <= 1).all(), f"Out of range [-1,1]: {gi.name}"
            assert torch.allclose(y1, y2, atol=0, rtol=0), f"Non-deterministic: {gi.name}"

        except Exception:
            # Log complet de la stack pour ce générateur, puis on remonte l'erreur.
            log.exception("Failure in generator %s", gi.name)
            raise
