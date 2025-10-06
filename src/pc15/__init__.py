"""PC15 — unified API (v15)
Install once, import one namespace:

    pip install "git+https://github.com/agaloppe84/ProceduralCodec-v15@v15.0.0"

Usage:

    import pc15 as pc
    y_out = pc.encode_y(img)["bitstream"]
    yhat  = pc.decode_y(y_out)

Or detailed modules:

    from pc15 import codec, proc, metrics, data, viz, wf
"""

__version__ = "15.0.0"

# Bring subpackages into a single namespace
# These imports assume the existing monorepo layout under packages/*/src
# If a subpackage is missing locally, we keep the namespace attribute None.
try:
    import pc15codec as codec
except Exception:
    codec = None
try:
    import pc15proc as proc
except Exception:
    proc = None
try:
    import pc15metrics as metrics
except Exception:
    metrics = None
try:
    import pc15data as data
except Exception:
    data = None
try:
    import pc15viz as viz
except Exception:
    viz = None
try:
    import pc15wf as wf
except Exception:
    wf = None
try:
    import pc15vq as vq
except Exception:
    vq = None
try:
    import pc15core as core
except Exception:
    core = None

# High-level convenience re-exports (top-level functions)
if codec is not None:
    try:
        from pc15codec import encode_y, decode_y, rans_encode, rans_decode, build_rans_tables, read_bitstream, write_bitstream, score_rd_numpy
    except Exception:
        # Not all low-level helpers are guaranteed stable; export what works
        from pc15codec import encode_y, decode_y  # type: ignore
else:
    encode_y = decode_y = None  # type: ignore

if proc is not None:
    try:
        from pc15proc import list_generators, render
    except Exception:
        list_generators = render = None  # type: ignore
else:
    list_generators = render = None  # type: ignore

if metrics is not None:
    try:
        from pc15metrics import psnr, ssim
    except Exception:
        psnr = ssim = None  # type: ignore
else:
    psnr = ssim = None  # type: ignore

if data is not None:
    try:
        from pc15data import scan_images, to_luma_tensor, build_manifest, ensure_symlink
    except Exception:
        scan_images = to_luma_tensor = build_manifest = ensure_symlink = None  # type: ignore
else:
    scan_images = to_luma_tensor = build_manifest = ensure_symlink = None  # type: ignore

if viz is not None:
    try:
        from pc15viz import montage, plot_rd, preview_tile_vs_synth
    except Exception:
        montage = plot_rd = preview_tile_vs_synth = None  # type: ignore
else:
    montage = plot_rd = preview_tile_vs_synth = None  # type: ignore

if wf is not None:
    try:
        from pc15wf import pc15_name, atomic_write, load_manifest, save_manifest, log_append
    except Exception:
        pc15_name = atomic_write = load_manifest = save_manifest = log_append = None  # type: ignore
else:
    pc15_name = atomic_write = load_manifest = save_manifest = log_append = None  # type: ignore

__all__ = [
    # sub-namespaces
    "codec", "proc", "metrics", "data", "viz", "wf", "vq", "core",
    # convenience
    "encode_y", "decode_y", "rans_encode", "rans_decode", "build_rans_tables",
    "read_bitstream", "write_bitstream", "score_rd_numpy",
    "list_generators", "render",
    "psnr", "ssim",
    "scan_images", "to_luma_tensor", "build_manifest", "ensure_symlink",
    "montage", "plot_rd", "preview_tile_vs_synth",
    "pc15_name", "atomic_write", "load_manifest", "save_manifest", "log_append",
    "__version__",
]
