# Black Forest Labs / FLUX Learnings for Satellite Latent Flow

## Bottom line

Black Forest Labs' FLUX work supports the direction of latent flow matching, but
it also makes the weakest part of our current direction very clear: the latent
space is not an incidental detail. The VAE/autoencoder is part of the model's
core capability. For our satellite task, we should train and evaluate a
domain-specific VAE before treating latent flow matching results as meaningful.

The practical takeaway is:

- Keep pixel-space flow matching as the reliable baseline.
- Treat SD-style generic VAEs as comparison points, not as the target design.
- Build a satellite VAE with a reconstruction ceiling high enough for roads,
  buildings, pools, vegetation, and color separation.
- Only then train the flow model in latent space.

## What Black Forest appears to be doing

### 1. FLUX is latent flow matching, not pixel flow matching

The public FLUX.1 model cards describe FLUX.1 as a 12B parameter rectified flow
transformer for text-to-image generation. The FLUX.1 Kontext report is more
explicit about the setup: it presents a generative flow matching model for image
generation and editing in latent space.

For our purposes, the architecture pattern is:

1. Encode RGB images into learned image latents.
2. Train a rectified-flow / flow-matching transformer over those latents.
3. Decode sampled latents back to RGB.

That means the generator does not need to model every pixel-level texture
directly, but it still depends completely on the autoencoder preserving the
visual facts that matter.

### 2. The latent representation is optimized, not assumed

BFL's FLUX.2 writeup says FLUX.2 uses a new VAE that provides the foundation for
the FLUX.2 flow backbones, and that they retrained the latent space to improve
learnability and image quality together. They frame this as a
learnability-quality-compression tradeoff.

That is the main lesson for us. A latent codec has three jobs:

- Compression: make the generative model cheaper by reducing spatial/token cost.
- Quality: reconstruct visually important details after decode.
- Learnability: present a latent distribution that the flow model can learn.

Average pooling mostly optimizes compression while sacrificing quality.
Pixel-unshuffle preserves quality but does not really learn a better
representation. A generic SD VAE may reconstruct natural images well, but it was
not trained to prioritize satellite-specific details like thin roads, roof
boundaries, small pools, field texture, or segmentation-aligned class edges.

### 3. Unified generation and editing come from conditioning as tokens/context

FLUX.1 Kontext handles text-to-image, local edits, global edits, style/object
reference, and iterative workflows in one model by conditioning on text and
image inputs together. The exact scale is far beyond this project, but the
pattern matters: the model sees conditioning as part of the sequence/context
rather than as a separate one-off path for each task.

For our satellite model, this argues for keeping the conditioning interface
simple and stable:

- semantic mask condition
- optional coordinate condition, only if it empirically helps
- future reference image / partial-image condition as an explicit input mode

We should avoid mixing inpainting, coordinates, and latent changes all at once.
The model should earn each conditioning path through ablations.

### 4. Distillation is a later optimization

FLUX.1 [dev] and [schnell] are distilled variants aimed at efficiency. That is
not the next step for us. We first need a high-quality teacher-like model path:
good VAE, stable latent flow training, and trustworthy metrics. Distillation only
matters once the baseline can generate good samples.

## What this implies for our satellite VAE

### We should train our own VAE

I agree with the hypothesis that we need a domain-specific VAE. The satellite
domain has visual priorities that generic image VAEs are unlikely to optimize:

- thin roads and paths
- hard roof/building edges
- small pools and water features
- vegetation texture without turning everything into green mush
- realistic class boundaries from semantic masks
- stable color separation between concrete, roofs, vegetation, water, and soil

If the VAE blurs these details, the latent flow model cannot recover them. The
flow model can only generate detail that exists in the latent representation or
that the decoder hallucinates from that representation.

### The VAE should be judged before the generator

Before any 20k or main training run, we should run a codec ceiling audit:

1. Take held-out validation RGB images.
2. Encode and decode them through the VAE.
3. Compare original vs reconstruction.
4. Report both aggregate and class-conditional metrics.

This gives us the best possible decoded image quality before training a latent
flow model. If the VAE ceiling is poor, a latent generator run is not worth the
H100 time.

Suggested reconstruction metrics:

- MSE / PSNR / SSIM for basic fidelity.
- Edge/gradient error for roads, roofs, and pool boundaries.
- Per-class reconstruction metrics using the semantic mask.
- Saturation/chroma/color histograms as telemetry only.
- Small-object retention for pools/buildings if masks can identify them.

Color telemetry should remain diagnostic. I would not make color matching a
training goal unless we define it as a low-level reconstruction term on paired
input/output images. We should avoid optimizing subjective color preferences.

### Start less compressed than we think

BFL's learnability-quality-compression framing suggests that too much
compression can make the flow model cheaper but destroy the target. For us, an
8x latent may be too aggressive at the start.

Recommended first VAE targets:

- Start with 4x downsample.
- Try latent channels 4, 8, and 16.
- Move to 8x only if the 4x model reconstructs well and training cost is too
  high.

The first useful result is not "smallest latent". It is "smallest latent whose
reconstructions preserve the structures we care about".

## Proposed implementation path

### Phase 1: VAE-only training

Add a native satellite autoencoder/VAE training path:

- Encoder/decoder: small convolutional architecture, probably with residual
  blocks and downsample/upsample stages.
- Latent bottleneck: configurable channels and downsample factor.
- Optional KL term, but keep it weak at first. Reconstruction quality matters
  more than sampling purity for the first experiment.
- Store latent mean/std from the training set for downstream normalization.
- Save VAE checkpoint independently from flow checkpoints.

Initial losses:

- L1 or Charbonnier reconstruction loss.
- MSE as a secondary stabilizer.
- SSIM or MS-SSIM if we want a structural reconstruction term.
- Gradient/edge loss to protect roads, roofs, and small boundaries.

Avoid for the first pass:

- generic perceptual losses that depend on an unrelated natural-image network
- adversarial loss
- color preference losses

Those may be useful later, but they add ambiguity when we are still trying to
understand the data and failure modes.

### Phase 2: VAE ceiling report

Create a reproducible report for each VAE candidate:

- `vae_recon_mse`
- `vae_recon_psnr`
- `vae_recon_ssim`
- `vae_recon_edge_mae`
- `vae_recon_by_class`
- `vae_recon_color_telemetry`
- visual grid of original / reconstruction / absolute error

This report should run without flow-model training.

Decision rule:

- If roads/buildings/pools are visibly blurred, reject the codec.
- If colors drift but structure is strong, keep it as a candidate and diagnose
  with telemetry.
- If both structure and color are strong, use it for latent flow.

### Phase 3: Latent flow matching with frozen VAE

Once a VAE passes the ceiling audit:

- Freeze the VAE.
- Encode training RGB crops to latents.
- Normalize latents using stored mean/std.
- Train the flow model on normalized latents.
- Downsample semantic masks to latent resolution in a way that preserves class
  fractions.
- Decode generated latents for RGB metrics and web samples.

Important: compare against pixel-space flow using the same data split, seeds,
training budget, and generation prompts/masks.

### Phase 4: Only then consider stronger tools

After the simple VAE path works, consider:

- adversarial decoder sharpening
- larger latent channels
- multi-scale decoder skips
- mask-aware reconstruction weighting
- distillation for faster inference

Those should come after the baseline proves the architecture, not before.

## How this changes how we interpret our current code

The current latent path is useful as a plumbing test. It verifies that config,
training, checkpoint metadata, inference, and decoded metrics can move through a
latent path. It should not be treated as a final latent-model design.

If the current path uses a generic SD VAE, it is a decent comparison baseline
but not the target. If it uses average-pool or pixel-unshuffle latents, it is
even more clearly an infrastructure test:

- average-pool latents are expected to be muddy
- pixel-unshuffle latents are an invertible control, not a learned compression
- generic SD VAE latents may be mismatched to satellite imagery

The BFL lesson is that "latent flow matching" is not just moving the same model
to a smaller tensor. It is jointly choosing a representation that the generator
can learn and the decoder can render faithfully.

## Recommended next PR

The next PR should be VAE-only, not a metrics grab bag and not another generator
change.

Scope:

- Add native satellite VAE model.
- Add VAE training command/config.
- Add VAE reconstruction audit command.
- Add artifact format for VAE checkpoint and latent stats.
- Add tests for config loading, encode/decode shape, checkpoint metadata, and
  deterministic reconstruction audit on tiny data.

Out of scope:

- latent flow generator training
- inpainting
- new conditioning modes
- adversarial losses
- distillation

Once that PR lands, the follow-up PR can add "train flow model with frozen
satellite VAE latents".

## Sources

- Black Forest Labs, [FLUX.2: Frontier Visual Intelligence](https://bfl.ai/blog/flux-2).
- Black Forest Labs, [FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space](https://arxiv.org/abs/2506.15742).
- Black Forest Labs, [FLUX.1 dev model card](https://huggingface.co/black-forest-labs/FLUX.1-dev).
- Black Forest Labs, [FLUX.1 schnell model card](https://huggingface.co/black-forest-labs/FLUX.1-schnell).
- Black Forest Labs, [official FLUX inference repository](https://github.com/black-forest-labs/flux).
- Dao et al., [Flow Matching in Latent Space](https://arxiv.org/abs/2307.08698).
