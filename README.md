# TinyVision Rust

Zero-dependency Rust image-model experiments designed for local CPU execution.

## Contents

- Standard-library-only Rust training and inference code.
- MNIST recognition baseline: 784 → 128 ReLU → 10-way softmax.
- CIFAR-10 classifier and low-resolution object-image sampler.
- Isolated Conv2D, SSM, and MoE validation tests.
- Trained `.tlm` artifacts, generated image samples, scripts, and presentation source.

## Verified results

- MNIST training accuracy: 99.04%.
- MNIST held-out test accuracy: 97.12% on 10,000 official test images.
- CIFAR-10 classifier held-out accuracy: 52.65% with the original spatially-blind baseline.
- CIFAR sampler: 32×32 RGB class exemplars for cat, automobile, and other CIFAR-10 classes.

## Important limitations

The current CIFAR object model is an exemplar sampler, not a novel GAN, VAE, or diffusion generator. It does not provide bounding-box object detection. A real detector requires annotated bounding boxes, and clear high-resolution generation requires a larger convolutional generator and substantially more compute.

## Build

```bash
cargo build --release
```

See the included READMEs and source files for model-specific commands.
