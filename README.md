# TinyVision Rust

Zero-dependency Rust image-model experiments designed for local CPU execution.

## Project announcement

I built TinyVision Rust as a transparent, local-first computer-vision project with no Rust runtime dependencies.

The project includes a from-scratch MNIST recognition model, CIFAR-10 classification experiments, a low-resolution class-conditioned image sampler, trained `.tlm` model files, and isolated validation tests for Conv2D gradients, SSM recurrence, and MoE routing.

The MNIST baseline achieved **97.12% accuracy on the official held-out test set**. The results are intentionally honest: the current CIFAR model is an exemplar sampler rather than a true GAN, VAE, or diffusion generator, and it does not yet perform bounding-box object detection.

## Contents

- Standard-library-only Rust training and inference code.
- MNIST recognition baseline: 784 → 128 ReLU → 10-way softmax.
- CIFAR-10 classifier and low-resolution object-image sampler.
- Isolated Conv2D, SSM, and MoE validation tests.
- Trained `.tlm` artifacts and generated image samples.

## Verified results

| Metric | Result |
|---|---:|
| MNIST training accuracy | 99.04% |
| MNIST held-out test accuracy | 97.12% on 10,000 official test images |
| CIFAR-10 held-out accuracy | 52.65% with the original spatially-blind baseline |
| CIFAR sampler output | 32×32 RGB class exemplars |

## Download and build

Install Rust from [rustup.rs](https://rustup.rs/), then verify:

```bash
rustc --version
cargo --version
```

Clone the private repository:

```bash
git clone https://github.com/sazalo101/tinyvision-rust.git
cd tinyvision-rust
```

The GitHub upload stores the Rust files at the repository root. Cargo normally expects them inside `src/`.

On Linux or macOS:

```bash
mkdir -p src
mv *.rs src/
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force src
Move-Item *.rs src/
```

Build the release binaries:

```bash
cargo build --release
```

The binaries are written to `target/release/`. Windows builds use the `.exe` suffix.

## Generate CIFAR-10 class images

CIFAR-10 class IDs are:

```text
0 airplane   1 automobile   2 bird   3 cat   4 deer
5 dog        6 frog         7 horse  8 ship  9 truck
```

Generate a cat using the included `cifar_objects.tlm` model:

```bash
./target/release/cifar_sampler generate cifar_objects.tlm 3 0 generated_cat.ppm
```

Generate an automobile:

```bash
./target/release/cifar_sampler generate cifar_objects.tlm 1 0 generated_car.ppm
```

Windows PowerShell:

```powershell
.\target\release\cifar_sampler.exe generate cifar_objects.tlm 3 0 generated_cat.ppm
.\target\release\cifar_sampler.exe generate cifar_objects.tlm 1 0 generated_car.ppm
```

The output is a 32×32 PPM image. Convert it to PNG with Pillow if desired:

```bash
python3 -c "from PIL import Image; Image.open('generated_cat.ppm').save('generated_cat.png')"
```

## Run MNIST image identification

For a 28×28 PGM digit image:

```bash
./target/release/mnist_image infer mnist.tlm digit.pgm
```

Windows PowerShell:

```powershell
.\target\release\mnist_image.exe infer mnist.tlm digit.pgm
```

## Retraining

The trained `.tlm` files are included for immediate local inference. The CIFAR-10 dataset itself is not included in the repository. To retrain, download the CIFAR-10 binary dataset separately and pass its directory to the training command documented in the corresponding Rust source.

## Important limitations

The current CIFAR object model is an exemplar sampler, not a novel GAN, VAE, or diffusion generator. It does not provide bounding-box object detection. Clear high-resolution generation requires a larger convolutional generator and substantially more compute. Real item detection also requires images with bounding-box annotations.

## License and status

This repository is an experimental, reproducible CPU vision project. The checkpoints and samples are included for local testing; benchmark values above are measured results, not claims of production-grade accuracy.

Repository: https://github.com/sazalo101/tinyvision-rust

#Rust #ComputerVision #MachineLearning #AI #OpenSource #CIFAR10 #MNIST
