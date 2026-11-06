# photoflow — batch replica of a Xiaomi 14T Pro photo-editor workflow

Python 3.11 CLI for native Windows 11 that post-processes hundreds of AI-generated portraits
(too smooth) with a fixed pipeline: **crop → skin texture → eyes → "AI Beautify" tone → NAT LUT → grain**.

> This folder lives in `xiaomi-photo-cli/` because the repo root is an unrelated Astro website with
> its own `README.md`, which I did not want to overwrite.

## Status: what is real, what is approximate

| Piece | Status |
|---|---|
| Crop, batch/multiprocessing, resumable manifest, metadata passthrough, JPEG q95 4:4:4 / PNG | Works, covered by tests |
| MediaPipe face landmarks, skin mask, iris/sclera masks | Works. Checked on a real portrait (scikit-image's astronaut), not on your images |
| Skin texture (frequency separation + procedural pores) | Works. **Look is a first guess**; amplitudes are my defaults, you must tune on your faces |
| Eyes (local contrast + sharpening, halo-free) | Works, same caveat |
| Tone ("AI Beautify") | Works as a model. **Slider-to-physics constants are placeholders** until you run `calibrate` |
| NAT LUT | **Placeholder = single-image estimate of NAT 100** (logged, flagged in the manifest), fitted to one original/NAT-100 pair you supplied. Hold-out ΔE76 ≈ 2.8-3.1 vs 5.6-6.4 untouched, *within that one scene*. Replace with a real Hald |
| `make-hald`, `lut-from-hald`, `calibrate` | Work; validated against a simulated phone with known ground truth (see "Calibration") |
| `contact-sheet`, `qc` | Work. QC thresholds are heuristics, re-threshold from the CSV numbers |
| Face-parsing ONNX | Code path works (tested with a tiny fake model). **No model URL shipped**, see decision 9 |
| ComfyUI img2img | Interface + one example workflow, tested against a mock server only. Never run against real ComfyUI |
| **Native Windows** | **Not run on Windows** (developed in a Linux container). Spawn-based multiprocessing (the Windows mode) is what the tests use, and all Windows wheels were verified to exist; the first Windows run is still the real test |

## Install (Windows 11, PowerShell, from this folder)

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1            # venv + deps + model download + tests
# optional GPU provider for the optional face-parsing ONNX:
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -Gpu dml    # DirectML (any GPU)   or   -Gpu cuda
.\.venv\Scripts\Activate.ps1
photoflow --help
```

Verified to exist: Windows cp311 wheels for `mediapipe`, `onnxruntime`, `onnxruntime-directml`,
`onnxruntime-gpu`, `opencv-contrib-python`, `scipy`. The Face Landmarker model is downloaded by
`photoflow fetch-models` from Google's public MediaPipe bucket (returned HTTP 200 during the build;
SHA-256 pinned in `photoflow/face.py`, a mismatch only warns).

## Commands

```powershell
# 1. Batch (resumable; rerun the same command after Ctrl+C or a crash)
photoflow run D:\ai\in D:\ai\out --preset heisenberg --jobs 4
photoflow run D:\ai\in D:\ai\out --dry-run          # show what would be done
photoflow run D:\ai\in D:\ai\out --force            # ignore manifest

# 2. Build the NAT LUT from the phone (see "Hald workflow")
photoflow make-hald hald_identity.png
photoflow lut-from-hald hald_nat50.png luts\nat.cube --baked-strength 50 --preset heisenberg

# 3. Calibrate tone + LUT blend from original/edited pairs (same file names in both folders)
photoflow calibrate D:\pairs\before D:\pairs\after --preset heisenberg

# 4. Review
photoflow contact-sheet D:\ai\in D:\ai\out --out sheet.jpg --n 12 --seed 1
photoflow qc D:\ai\out --out qc_report.csv --jobs 4

# helpers
photoflow fetch-models
photoflow make-synthetic demo\synthetic_in --n 6
```

### Hald workflow (this is where it is easy to get it wrong)

1. `photoflow make-hald hald_identity.png` → 512×512 identity Hald (level 8, 64³ cube).
2. On the phone: open it in the stock editor, apply **only** NAT (no AI Beautify, no crop, no auto-anything), export at full size. **PNG if the editor lets you; JPEG works but add `--smooth 0.8`** to `lut-from-hald`, otherwise compression noise ends up in the LUT.
3. `lut-from-hald hald_nat50.png luts\nat.cube --baked-strength 50 --preset heisenberg`

**Double-application trap.** You plan to process the Hald at NAT 50 *and* blend the LUT at 50%. That
would give an effective 25%. So the preset has `lut_nat.lut_baked_strength`: the strength already inside
the Hald. The pipeline blends `strength / lut_baked_strength` (50/50 = 1.0 → LUT used fully, which equals
NAT 50). If you process the Hald at NAT 100 instead, set baked 100 and the 50% blend does the work.
**Recommendation: bake at 100** — you can then pick any strength on the PC without a new trip to the phone.

A Hald only captures a *global color mapping*. If NAT is adaptive (content/face-aware, local contrast),
that part cannot be reproduced by any LUT; `calibrate` residuals will show it.

### Calibration

Provide 20+ varied photos (different skin tones, exposures) with the **same file names** in both folders.
On the phone apply exactly: AI Beautify 66 (exposure +3, brightness +15, contrast +15) → NAT 50, export, no crop
(pairs whose aspect ratio differs are skipped and reported). The model mirrors the pipeline order
(tone, then LUT) and fits `EV`, `brightness offset`, `contrast gain` and the LUT blend by robust least squares
in Lab, and prints ΔE76 for: no processing / current preset / fitted. It then writes the fitted per-point
constants and `lut_nat.strength` into the preset **in place, keeping comments** (a `.bak` copy is made).

Validation: I simulated a phone (known tone recipe + a different LUT baked into a Hald at 50%) and ran
the whole chain. ΔE76 fell from **9.3 (no processing) → 2.0 (preset guesses) → 0.4 (fitted)**, below the 8-bit
noise floor of the data. Honest limit: exposure, brightness and contrast overlap on the tone curve, so
from one recipe only their *combined* effect is identified. The individual constants did **not** recover
the simulated truth (fitted EV 0.064 vs true 0.09, gain 1.096 vs 1.08) even though the output matches.
A weak prior toward the preset breaks ties. If you later change slider values, results extrapolate
approximately. For separable constants, add pairs edited one slider at a time.

### Reference pair findings (original vs NAT 100, 1280×943, both sRGB-tagged)

- NAT 100 is **desaturating and cooler**, not a punchy filter: chroma ×0.76-0.83 in every region (foliage 0.76, skin 0.80-0.83, hair 0.77), b\* 16.0 → 12.3, blacks lifted (1st percentile L 1.4 → 3.4), lower mid-tones darker (25th percentile 36.8 → 32.8).
- My first placeholder (S-curve, +4% saturation, warm) did the **opposite**: ΔE76 to the phone result was 9.88, *worse than doing nothing* (5.86). It was replaced by the fit above.
- The chroma scale was stable across three independent spatial splits (0.78-0.79); the lightness curve is the least certain part.
- One image can never give a full 3D LUT (the colour cube is mostly unobserved here). It also cannot tell you whether NAT is adaptive. Remaining error ≈ ΔE 2.8 is whatever a global model misses.

## Pipeline & preset

`presets/heisenberg.yaml` holds your values; every step has `enabled` and `strength` 0-100
(100 = the step's nominal parameters, all tunable in the YAML or `photoflow/config.py` defaults).

| Step | Your value | What it does |
|---|---|---|
| `crop` | 4% per side | p% of width off left+right and p% of height off top+bottom (aspect preserved) |
| `skin_texture` | 100 | Mask = face oval (eroded) minus eyes, brows, lips, each feathered; radii are fractions of face width. Luminance of Lab only: soft-limited boost of fine+medium detail, plus procedural micro-relief and pore dots (band-passed noise, dark lobes), weighted toward midtones and toward smooth areas. Tiny chroma mottling. Seeded per file |
| `eyes` | 100 | Iris (disk ∩ eyelid hull) and sclera (hull minus iris, eroded). Local contrast + fine sharpening against a *masked* blur per region (no iris→sclera halo), soft-limited, clamped to the local min/max of the original (no overshoot). Sclera gets a third (local contrast) to a half (sharpening) of the iris gain, to avoid a "creepy eyes" look |
| `tone` | 66 | exposure +3, brightness +15, contrast +15, scaled by 0.66 → EV, display-space offset, gain about a pivot (formulas in `tone.py`). Mapping constants in the YAML, set by `calibrate` |
| `lut_nat` | 50 | `.cube` or Hald PNG, trilinear, blended per the baked-strength rule above |
| `grain` | off | luminance grain, midtone-weighted |
| `img2img` | off | optional ComfyUI pass, runs right after crop (before texture, so the texture is not diffused away) |

Optional per-photo jitter (`jitter.enabled: true`): small random variation of each step strength and of
the three tone slider values, **deterministic from the file hash** so reruns reproduce exactly.

### Optional ComfyUI low-denoise pass

`steps.img2img.enabled: true`, set `checkpoint` (an SD1.5/SDXL checkpoint name you actually have; I cannot
verify yours) and keep ComfyUI running on `127.0.0.1:8188`. `workflows/img2img_lowdenoise_example.json`
is an API-format workflow with `__CKPT__ __IMAGE__ __POSITIVE__ __NEGATIVE__ __DENOISE__ __SEED__` placeholders.
Images are downscaled to `max_side` (default 1024, to fit 8 GB VRAM), processed, resized back and
blended by strength; skin texture then runs on top. ComfyUI serializes jobs on one GPU, so extra `--jobs` do not speed this up.

## Decisions I made (so you do not have to be asked)

1. **Location**: subfolder, not repo root (see top).
2. **Dependencies**: `opencv-contrib-python` rather than `opencv-python` (MediaPipe depends on it; both would clash). **`scipy` added** to your stack, only for calibration's `least_squares`.
3. **Metadata**: EXIF/ICC/XMP bytes are copied untouched; nothing added (no Software tag, no camera EXIF). One forced exception: if EXIF Orientation ≠ 1, pixels are rotated upright and the tag is set to 1, otherwise viewers rotate twice (face detection also needs upright). Side effects you may notice: EXIF dimension tags and any embedded EXIF thumbnail still describe the *original*, not the cropped result, since you asked for as-is.
4. **Pixels**: 8-bit output; color math in the file's encoded (sRGB-like) values, ICC profile passed through, no gamut conversion. 16-bit input is reduced to 8-bit. RGBA is kept (forces PNG). HEIC is not supported (no `pillow-heif`).
5. **No face** (`face.on_no_face`): default `global_only` (crop/tone/LUT/grain still run, entry flagged `no_face` in the manifest/log); `skip` produces no output. Multiple faces: all (up to 4) are processed.
6. **Resumability**: `<out>\.photoflow\manifest.jsonl`, written only by the parent, fsynced per image. Skip needs same input SHA-256, same config hash (effective preset + LUT bytes + pipeline version) and existing output. Errors are retried. Output is written to `*.tmp` then `os.replace`. OUTPUT may be inside INPUT.
7. **Windows multiprocessing**: `spawn` always, MediaPipe created once per worker, OpenCV threads = 1 per worker, default workers = half the cores. MediaPipe Face Landmarker runs on CPU (XNNPACK); your GPU is only used by the optional ONNX parser and ComfyUI.
8. **QC**: measured on a 1024 px copy. `blur` = Laplacian variance of the 256 px face crop; `excess_texture` = robust (MAD) noise sigma outside the face (hair + background, no hair segmentation) above a threshold, or far above skin noise; no face / multiple faces from the landmarker. All raw numbers are in the CSV. Heuristic: a legitimately textured background can flag.
9. **Face-parsing ONNX**: optional (`face.parsing_onnx`). I could not verify any hosted model (huggingface.co was blocked from my build environment), so **I did not invent a URL**. Bring any BiSeNet-style model with the CelebAMask-HQ 19-class layout (512 input, ImageNet mean/std; skin = labels 1 and 10; my recollection of that layout is [likely], not verified against a real model). It is intersected with the landmark mask. Providers: CUDA → DirectML → CPU.
10. **Placeholder LUT**: a 5-parameter global Lab model (lightness curve, chroma ×0.79 on a\* and b\*, tiny offsets) fitted to one reference pair, see "Reference pair findings". Warned in the console, in the log, and as `placeholder_lut` in the manifest notes. Treat output made with it as provisional; after you install the real LUT the config hash changes and everything re-processes automatically.

## Known weak spots (read before trusting 500 images)

- **Procedural pores are isotropic noise.** Real skin texture varies by region (forehead vs nose vs cheeks) and follows lighting. At full strength it can read as fine uniform "sand" on very large faces; reduce `skin_texture.strength` or `pore_amplitude`/`micro_amplitude` and look at 100% crops, not the contact sheet.
- **Texture is a mask, not a segmentation of semantics**: the oval hull follows landmarks, so hairline strands over the forehead and beards can receive texture unless you add the parsing model.
- **Defaults are guesses** (all `[Guessing]`): tone mapping constants, texture amplitudes, QC thresholds. `calibrate` fixes the tone/LUT part only.
- A global model cannot reproduce local components of the phone's AI Beautify (face smoothing, auto-HDR). `calibrate` warns when the residual is high (> ΔE 4).

## Tests

```powershell
pytest -q        # 32 pass, 1 skipped (real-face test needs .\samples and the model)
```

Synthetic-image tests cover: exact crop sizes, non-empty and correctly-shaped masks (eyes/brows/lips excluded, cheeks covered,
radii scale with face size), texture confined to the mask, eye changes confined and overshoot-free, tone scaling, LUT identity /
blend / Hald+cube round trips / baked-strength rule, idempotent resumable batch (unchanged inputs → zero work and byte-identical
outputs; changed input, deleted output, changed preset → only the right files redone), truncated manifest, serial vs spawn-parallel
equality, bad-file isolation, JPEG 4:4:4 q95, EXIF passthrough, calibration recovery, jitter determinism, QC flags,
ComfyUI client vs a mock server, ONNX parsing vs a tiny fake model. Tests use a fixed-layout detector so they need no model and no real face;
`tests/test_real_face.py` uses `.\samples` when present.

## Demo

`.\samples` did not exist, so `demo\` was generated from synthetic cartoon portraits:
`demo\contact_sheet.jpg` and `demo\qc_report.csv` (the two noisy-background images are flagged `excess_texture`, as intended).
These show the plumbing, **not** result quality. Judge quality on your own portraits at 100% zoom.

## Layout

```
photoflow/   cli, batch, pipeline, skin, eyes, tone, lut, calibrate, qc, contact, comfy, masks, face, parsing, imageio, config, yamledit, synthetic
presets/heisenberg.yaml    workflows/img2img_lowdenoise_example.json    models/ (landmarker, downloaded)    tests/    demo/
```
