# CUDA MNIST Classifier

A fully-connected neural network for MNIST digit classification, with forward and
backward propagation implemented as hand-written CUDA kernels (no cuBLAS, no
deep learning framework). Originally built as a GPU programming assignment at
Centrale Nantes and deployed on a Jetson Nano provided by the school.

The project started from a sequential CPU reference implementation (the
assignment's starting point) and was rewritten so that every matrix operation
in the training loop — matrix multiply, element-wise add/subtract/Hadamard
product, transpose, scalar multiply, and the sigmoid activation and its
derivative — runs as a custom CUDA kernel.

## Architecture

A 3-layer multilayer perceptron, trained with minibatch gradient descent:

```
Input (784 = 28x28 pixels) -> Hidden (30, sigmoid) -> Output (10, sigmoid)
```

- **Loss / training signal**: per-pixel sigmoid activations, weights updated
  with backpropagation (`backward` in `ann.cu`), no explicit loss function
  object — the delta rule is derived directly from `(activation - target)`.
- **Optimizer**: plain minibatch SGD, learning rate `alpha = 0.05`, minibatch
  size 16 (both set in `main.cu`).
- **Weight init**: Gaussian (`normalRand`, Box-Muller transform), scaled by
  `1/sqrt(fan_in)`.

### CUDA design

Each matrix (`matrix_t` in `matrix.h`) keeps a host pointer (`m`) and a device
pointer (`m_gpu`) side by side. Transfers between the two are explicit and
intentionally rare:

- Inputs and labels are copied to the GPU **once per minibatch**.
- The entire forward and backward pass (`forward`/`backward` in `ann.cu`)
  chains CUDA kernel launches on `m_gpu` buffers with no intermediate
  host round-trip.
- Results are only copied back to the host when accuracy needs to be read
  on the CPU (`matrix_GPU_to_CPU` in the `accuracy` function in `main.cu`).

All scratch matrices a layer needs during forward/backward (`z1_tmp`,
`z2_tmp`, `tw`, `delta_tmp`, `dfz`, `ta`, `w1`, `b1`, ...) are pre-allocated
once when the network is built (`create_layer` in `ann.cu`), so the training
loop itself does zero `cudaMalloc` calls.

Kernels are intentionally simple (one thread per output element, 32-thread
1D blocks for element-wise ops, 32x32 2D blocks for matrix multiply/transpose)
— there is no shared-memory tiling or use of cuBLAS. See
[Possible improvements](#possible-improvements) for what a faster version
would look like.

Every CUDA Runtime call is wrapped in `CHECK_ERROR` (`err.h`), which prints
the CUDA error string, file, and line, and aborts on failure.

## Repository structure

| File | Purpose |
|---|---|
| `main.cu` | Training loop, MNIST loading glue, accuracy evaluation |
| `ann.cu` / `ann.h` | Network/layer structures, forward/backward propagation |
| `matrix.cu` / `matrix.h` | Matrix type, CUDA kernels, and CPU reference implementations |
| `mnist.cu` / `mnist.h` | Parser for the original MNIST IDX file format |
| `test.cu` | Minimal scratch program to sanity-check individual GPU kernels |
| `err.h` | `CHECK_ERROR` macro for CUDA Runtime error checking |
| `Makefile` | Build rules (`nvcc`), configurable target GPU architecture |

`matrix.cu` also contains sequential, host-only versions of every matrix
operation (`matrix_sum`, `matrix_dot`, `hadamard_product`, `matrix_minus`,
`matrix_transpose`, `matrix_scalar`). These are the original pre-CUDA baseline
the assignment started from; the training pipeline in `ann.cu` only calls the
`*_gpu` kernels, so today these CPU functions are kept for reference rather
than wired into a runnable CPU-only path (see below).

## Building and running

### Requirements

- An NVIDIA GPU + CUDA Toolkit (`nvcc`).
- The MNIST dataset, as the original 4 uncompressed IDX files, in the working
  directory: `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`,
  `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`.

Download and decompress them (a mirror that doesn't require auth, unlike
the original Yann LeCun site which has had availability issues):

```bash
base=https://storage.googleapis.com/cvdf-datasets/mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte; do
  curl -O "$base/$f.gz" && gunzip "$f.gz"
done

# main.cu expects a dot before "idx", not a hyphen — rename accordingly:
mv train-images-idx3-ubyte train-images.idx3-ubyte
mv train-labels-idx1-ubyte train-labels.idx1-ubyte
mv t10k-images-idx3-ubyte  t10k-images.idx3-ubyte
mv t10k-labels-idx1-ubyte  t10k-labels.idx1-ubyte
```

### Build

```bash
make                 # builds ./ann, targets sm_53 (Jetson Nano) by default
make ARCH=sm_75       # e.g. Colab/Kaggle T4 (Turing)
make ARCH=sm_86       # e.g. RTX 30xx (Ampere)
make test            # builds the small kernel sanity-check program
```

### Run

```bash
./ann
```

Prints the starting (random-weights) accuracy, then per-epoch test accuracy
as it trains. The number of epochs is currently fixed at 1 in `main.cu`
(`for (int epoch = 0; epoch < 1; epoch++)`) — increase it to train longer.

### Profiling

```bash
make profile          # rebuilds ./ann with gprof instrumentation (-pg)
./ann
gprof ann gmon.out | less
```

`gprof` only instruments host (CPU) code — it will show time spent in
`main`, `accuracy`, kernel *launch* overhead, etc., but not what happens
inside the GPU kernels themselves. To profile the kernels, use
`nvprof`/Nsight Systems or Nsight Compute on a machine with the corresponding
NVIDIA tools installed.

#### Profiling the GPU kernels on Colab

Colab's CUDA image ships with `nvprof`, `nsys` (Nsight Systems), and `ncu`
(Nsight Compute) preinstalled, so kernel-level profiling works out of the box
on the free T4 runtime:

```bash
!make ARCH=sm_75                # T4 is Turing, compute capability 7.5

# Legacy, simplest: per-kernel time/call counts, straight to stdout.
# nvprof is deprecated from Volta/Turing onward (removed entirely on
# Ampere+), but still works on the T4.
!nvprof ./ann

# Recommended: timeline of kernels + memcpys, download the .nsys-rep
# and open it in the (free) Nsight Systems desktop app.
!nsys profile -o ann_profile ./ann
!nsys stats ann_profile.nsys-rep

# Per-kernel metrics (occupancy, memory throughput, ...).
!ncu --kernel-name matrix_dot_kernel ./ann
```

Add `-lineinfo` to `NVCCFLAGS` (`make ARCH=sm_75 NVCCFLAGS="-O3 -arch=sm_75 -lineinfo -lm"`)
to get source-line correlation in the Nsight reports. None of this needs
extra `cudaDeviceSynchronize()` calls in the code — these tools hook into the
CUDA driver directly, unlike host-side wall-clock timing.

## Jetson Nano deployment

Developed and run on an **NVIDIA Jetson Nano** (Maxwell GPU, 128 CUDA cores,
compute capability `sm_53`), JetPack 4.6 / CUDA 10.2. `ARCH=sm_53` is the
Makefile default specifically so it builds out of the box on the Nano without
any extra flags:

```bash
make
./ann
```

## Can this run without a (local) GPU?

Short answer: the kernels in this repo (`matrix_dot_kernel`,
`matrix_sum_kernel`, etc.) are real CUDA device code — they need an actual
NVIDIA GPU at runtime. There is no practical CPU emulation mode for modern
CUDA: NVIDIA removed device emulation after CUDA 3.1, and third-party
CUDA-on-CPU projects (e.g. GPU Ocelot) are unmaintained and don't support
recent CUDA versions.

The practical way to run this without owning a GPU is to use a **free cloud
GPU**, which is a real GPU, not an emulation:

- **Google Colab** (free tier, NVIDIA T4): `Runtime > Change runtime type >
  GPU`, upload the repo, then from a notebook cell:
  ```bash
  !nvcc -O3 -arch=sm_75 -o ann main.cu ann.cu matrix.cu mnist.cu -lm
  !./ann
  ```
  (Colab images ship with the CUDA Toolkit preinstalled, so no setup beyond
  enabling the GPU runtime is needed.)
- **Kaggle Notebooks** (free tier, T4/P100) works the same way.
- Any cloud VM with an NVIDIA GPU (AWS/GCP/Azure/Lightning AI, etc.) — build
  with `make ARCH=<compute capability of that GPU>`.

A true CPU-only build is possible in principle, since `matrix.cu` already
contains sequential equivalents of every kernel (see the table above) — but
`ann.cu`'s `forward`/`backward` call the `*_gpu` versions exclusively, so
wiring a CPU training path would mean writing a CPU-only `forward`/`backward`
and gating `alloc_matrix`'s `cudaMalloc` calls behind a build flag. That's
listed as a possible improvement below rather than done here, since it
wasn't part of the original assignment scope.

## Results

Exact numbers depend on the random weight initialization and the run
environment — fill in your own after running `./ann`. As a reference point,
this architecture (1 hidden layer of 30 sigmoid units, minibatch SGD,
`alpha=0.05`) typically goes from ~10% accuracy (random weights, 10 classes)
to a meaningfully higher test accuracy after a single epoch over the 60k
training images; train for more epochs (raise the loop bound in `main.cu`)
to push it further.

## Possible improvements

- Wire the existing CPU reference matrix functions into an actual CPU-only
  build mode (useful for environments without any GPU access, and as a
  CPU vs. GPU correctness/performance comparison).
- Replace the naive matrix-multiply kernel with a shared-memory tiled
  version, or call cuBLAS (`cublasDgemm`) and compare throughput.
- Add `cudaDeviceSynchronize`/CUDA events around kernel launches to get
  accurate GPU-side timings instead of relying on host-side wall clock.
- Try `float` instead of `double` for the matrices — Jetson Nano's Maxwell
  GPU has much higher throughput in single precision.
- Add more hidden layers / units and compare accuracy vs. training time.

## License

MIT — see [LICENSE](LICENSE).
