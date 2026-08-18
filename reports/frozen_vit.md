# Frozen pretrained ViT: transfer learning, properly tested

**Result: macro precision 0.501 against 0.660 for a 5 k-parameter model built
for this data. ET precision 0.332 against 0.685.**

## Why this test exists

The earlier ViT result in this project (AUC 0.540, chance) used **random
weights**, because ImageNet checkpoints cannot be downloaded in this
environment. That measured a random projection and said nothing about transfer
learning. It should not have been left standing as if it had.

The checkpoint was supplied manually and assembled from three chunks: 152
tensors, fp16, loading into `torchvision.vit_b_16` with **0 missing and 0
unexpected keys**.

## Setup

* backbone **frozen**: 85.8 M parameters, **0 trainable**
* trainable: a single linear head, **2,307 parameters** -- inside the 1e3-1e4
  band where every model on this cohort peaks, so this does not contradict the
  capacity finding, which penalises *trainable* parameters
* input: 64x64 log spectrogram -> per-image min-max to [0,1] -> bilinear resize
  to 224 -> repeated to 3 channels -> ImageNet normalisation -> 768-d embedding
* merged cohort n=404 (49 ET), mixed protocol, validation-tuned priors, 20 splits

## Result

| model | trainable params | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|---|
| frozen ViT-B/16 + linear head | 2,307 (of 85.8 M) | 0.618 | 0.554 | 0.332 | 0.501 | 0.473 |
| logreg on 10 descriptors | 33 | ~0.64 | ~0.64 | ~0.58 | 0.619 | 0.580 |
| **two-stream (multitaper + IF trajectory)** | ~5,000 | **0.639** | **0.655** | **0.685** | **0.660** | **0.593** |

**85.8 M ImageNet parameters underperform ten numbers computed from a Welch
spectrum.**

It is not at chance (0.501 against ~0.33 for three classes), so ImageNet
features carry *some* signal -- edges and intensity gradients are still edges in
a spectrogram. Just far less than features built for tremor.

## Why it fails

ImageNet filters encode object parts, textures and natural-image statistics. A
tremor spectrogram is close to the opposite: one bright horizontal band on a
dark field, no objects, no texture, and the discriminative content is the
*position* and *sharpness* of that band. Almost nothing the backbone was trained
to detect is present, and the things that matter here it was never trained to
measure.

## What this closes

Combined with ResNet18 and WideResNet from scratch, `vit_b_16` with random
weights, `SpectrumTransformer` and `CrossStreamAttention`, this is the properly
executed version of "would a big pretrained model help". **It would not.** The
question can be considered settled rather than left open as an untested
possibility -- which is a materially stronger statement for a write-up.

Reproduce: `python -m experiments.frozen_backbone --weights vit_fp16.pt`
(rebuild the checkpoint with `cat vit_chunk_0* > vit_fp16.pt`).

## Small attention on the current input: also no

The frozen-ViT test above rules out a large pretrained transformer. This rules
out small ones, on the input the current model actually uses (16 log-bins + IF
trajectory, 404 patients) rather than the 61-bin raw spectra and 25 patients the
earlier `BilateralAttention` test used.

| model | params | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|---|
| **TwoStreamNet CNN (current best)** | 0.7 k | 0.639 | 0.655 | **0.685** | **0.660** | 0.593 |
| + SpectrumTransformer | 17.3 k | 0.625 | 0.655 | 0.665 | 0.648 | 0.595 |
| CrossStreamAttention | 4.9 k | 0.645 | 0.633 | 0.646 | 0.641 | 0.568 |

Paired against the current best, 20 splits:

| | macroP | macroF1 |
|---|---|---|
| SpectrumTransformer | -0.011 [-0.036, +0.010] | +0.002 [-0.018, +0.020] |
| CrossStreamAttention | -0.018 [-0.048, +0.013] | **-0.026 [-0.044, -0.007]** * |

`CrossStreamAttention` was the best-motivated variant -- spectrum bins query the
IF trajectory so the two streams condition on each other instead of meeting only
at the classifier head, which is the mechanism the published bilateral-wrist
result uses across limbs. It is significantly worse on macro F1.

## The attention question, closed

| form | result |
|---|---|
| frozen pretrained ViT-B/16 + linear head | macroP 0.501 |
| ViT / ResNet18 / WideResNet, random weights | at chance |
| `SpectrumTransformer` (17 k, current input) | -0.011, n.s. |
| `CrossStreamAttention` (5 k, current input) | -0.026 macroF1 * |
| `BilateralAttention` (25 patients, old input) | at chance |
| `AttnPoolBiLSTM` | +0.012 for the BiLSTM only; does not transfer to the TCN |

Attention does not help this problem at this scale, in pretrained or small form,
on old or current inputs. A 0.7 k-parameter 1-D convolution over the frequency
axis remains the best spectrum encoder measured.
