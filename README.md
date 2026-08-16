# TinyVision Rust

Zero-dependency Rust image-model experiments designed for local CPU execution.

## Project announcement

TinyVision Rust is a transparent, local-first computer-vision project implemented with the Rust standard library and no external Rust runtime dependencies.

The project includes a from-scratch MNIST recognition model, CIFAR-10 classification experiments, a low-resolution class-conditioned image sampler, trained `.tlm` model files, and isolated validation tests for Conv2D gradients, SSM recurrence, and MoE routing.

The MNIST baseline achieved **97.12% accuracy on the official held-out test set**. The results are intentionally honest: the current CIFAR model is an exemplar sampler rather than a true GAN, VAE, or diffusion generator, and it does not yet perform bounding-box object detection.

## Verified results

| Metric | Result |
|---|---:|
| MNIST training accuracy | 99.04% |
| MNIST held-out test accuracy | 97.12% on 10,000 official test images |
| CIFAR-10 held-out accuracy | 52.65% with the original spatially-blind baseline |
| CIFAR sampler output | 32×32 RGB class exemplars |

## Kali Linux: download and build

Install Rust and Cargo if they are not already installed:

```bash
sudo apt update
sudo apt install -y cargo rustc
rustc --version
cargo --version
```

Clone the public repository:

```bash
git clone https://github.com/sazalo101/tinyvision-rust.git
cd tinyvision-rust
```

The uploaded Rust files are stored at the repository root, while Cargo expects binary source files inside `src/`. Move them before building:

```bash
mkdir -p src
for f in *.rs; do
  [ -e "$f" ] && mv "$f" src/
done
```

Build the MNIST image utility:

```bash
cargo build --release --bin mnist_image
ls -lh target/release/mnist_image
```

The expected executable is:

```text
target/release/mnist_image
```

## Create a test PGM image

The `mnist_image` utility expects a **28×28 grayscale PGM** file. The following creates a simple synthetic digit-like image without additional Python packages:

```bash
cat > make_digit.py <<'PY'
w = h = 28
img = [[0 for _ in range(w)] for _ in range(h)]

def rect(x0, y0, x1, y1):
    for y in range(max(0, y0), min(h, y1 + 1)):
        for x in range(max(0, x0), min(w, x1 + 1)):
            img[y][x] = 255

# Thick handwritten-style test shape intended to resemble 5
rect(7, 4, 20, 7)
rect(6, 5, 9, 13)
rect(7, 11, 19, 14)
rect(17, 12, 20, 22)
rect(7, 20, 19, 23)

with open("digit.pgm", "wb") as f:
    f.write(b"P5\n28 28\n255\n")
    f.write(bytes(v for row in img for v in row))
PY

python3 make_digit.py
file digit.pgm
```

Expected file validation:

```text
digit.pgm: Netpbm image data, size = 28 x 28, rawbits, greymap
```

## Run MNIST image identification

Download the trained checkpoint if it is not already present:

```bash
[ -f mnist.tlm ] || curl -L --fail -o mnist.tlm \
  https://raw.githubusercontent.com/sazalo101/tinyvision-rust/main/mnist.tlm
```

Run inference with explicit paths:

```bash
./target/release/mnist_image infer ./mnist.tlm ./digit.pgm
```

A verified Kali run produced:

```text
predicted=4 confidence=100.00%
```

The synthetic image was intended by its drawing code to resemble a 5, but the model predicted 4. This is a valid end-to-end execution test, not a claim that the synthetic drawing was classified correctly. For a fair accuracy test, use real MNIST test images with known labels.

## Generate CIFAR-10 class images

CIFAR-10 class IDs are:

```text
0 airplane   1 automobile   2 bird   3 cat   4 deer
5 dog        6 frog         7 horse  8 ship  9 truck
```

Build the sampler:

```bash
cargo build --release --bin cifar_sampler
```

Generate a cat:

```bash
./target/release/cifar_sampler generate cifar_objects.tlm 3 0 generated_cat.ppm
```

Generate an automobile:

```bash
./target/release/cifar_sampler generate cifar_objects.tlm 1 0 generated_car.ppm
```

The outputs are 32×32 PPM images. Convert one to PNG with Pillow if desired:

```bash
python3 -c "from PIL import Image; Image.open('generated_cat.ppm').save('generated_cat.png')"
```

## Windows PowerShell equivalents

```powershell
New-Item -ItemType Directory -Force src
Move-Item *.rs src
cargo build --release --bin mnist_image
.\target\release\mnist_image.exe infer .\mnist.tlm .\digit.pgm
```

## Important limitations

The current CIFAR object model is an exemplar sampler, not a novel GAN, VAE, or diffusion generator. It does not provide bounding-box object detection. Clear high-resolution generation requires a larger convolutional generator and substantially more compute. Real item detection requires images with bounding-box annotations.

The MNIST utility classifies one 28×28 image into one digit class. It does not detect multiple objects, locate objects, or guarantee a correct prediction for every arbitrary drawing.

## Repository

https://github.com/sazalo101/tinyvision-rust

#Rust #ComputerVision #MachineLearning #AI #OpenSource #CIFAR10 #MNIST
