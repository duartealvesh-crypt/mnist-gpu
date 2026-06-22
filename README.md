# CUDA MNIST Classifier

> A 784–30–10 MLP for MNIST with forward/backprop written as **hand-rolled CUDA
> kernels** — no cuBLAS, no framework. Built for a GPU programming course at
> Centrale Nantes and run on a **Jetson Nano**.

The network is deliberately simple. The interesting part is the **iterative GPU
memory optimization**: each step was profiled with `nvprof`, and the dominant
matmul kernel went from **1.74 ms → 333 µs per call** by changing *how* memory
moves between host and device — not the math.

## Performance journey

All stages run on the Jetson Nano (Maxwell, sm_53), minibatch size 16 held
constant so timings stay comparable. `matrix_dot` is the hotspot (~90% of GPU
time throughout), so it's the metric tracked here.

| # | Memory strategy | `matrix_dot` / call | Dominant cost | Outcome |
|---|---|---|---|---|
| ① | GPU `cudaMalloc` + HtoD/DtoH copy **on every call** | 1.74 ms | host⇄device transfers | GPU as slow as CPU (~48 s) |
| ② | Unified (managed) memory | 1.97 ms | managed-memory sync | simpler code, still transfer-bound |
| ③ | **All ops on GPU**, weights resident — but still one CPU→GPU upload per epoch | 611 µs | residual per-epoch transfer | big drop |
| ④ | **Fully GPU-resident**: no host round-trip once the network is instantiated | **333 µs** | matmul kernel | matmul total 23.7 s → 12.9 s |

**Starting point (CPU baseline, gprof on an Intel i5-1355U):** `matrix_dot`
alone was ~90% of runtime (~55 s) — which is what motivated moving it to the GPU
in the first place.

**Takeaway:** on a memory-constrained device, naively offloading a kernel can be
*slower* than the CPU because of transfer overhead. The real win was making the
network fully GPU-resident — weights and activations stay on the device, and
after instantiation the training loop never copies back to the host — so kernels
chain with no round-trips. The kernel code barely changed; *where the memory
lives* did.

## How it works

3-layer MLP, minibatch SGD, sigmoid activations:

```
Input (784 = 28×28) → Hidden (30) → Output (10)
```

- **Optimizer:** plain minibatch SGD, `alpha = 0.05`, batch size 16 (`main.cu`).
- **Weight init:** Gaussian (Box-Muller), scaled by `1/sqrt(fan_in)`.
- **Memory model:** each `matrix_t` holds a host pointer (`m`) and a device
  pointer (`m_gpu`). All per-layer scratch buffers are allocated **once** at
  network creation (`create_layer`), so the training loop does zero `cudaMalloc`.
  Inputs/labels are copied to the GPU once per minibatch; `forward`/`backward`
  then chain kernel launches entirely on device memory (`ann.cu`).
- **Correctness check:** with this minimal config the model reaches ~60% test
  accuracy after a single epoch — the author's threshold for "the kernels are
  correct."

Every CUDA Runtime call is wrapped in `CHECK_ERROR` (`err.h`).

## Repository layout

| File | Purpose |
|---|---|
| `main.cu` | Training loop, MNIST loading, accuracy evaluation |
| `ann.cu` / `ann.h` | Layer structures, forward/backward propagation |
| `matrix.cu` / `matrix.h` | Matrix type, CUDA kernels (+ CPU reference versions) |
| `mnist.cu` / `mnist.h` | MNIST IDX file parser |
| `test.cu` | Standalone kernel sanity check |
| `Makefile` | `nvcc` build, configurable GPU arch |

`matrix.cu` keeps sequential CPU versions of every op (`matrix_dot`,
`matrix_sum`, …) — the pre-CUDA baseline. The training path only calls the
`*_gpu` kernels; the CPU versions are kept for reference.

## Build & run

Requires an NVIDIA GPU + CUDA Toolkit (`nvcc`) and the four MNIST IDX files in
the working directory:

```bash
base=https://storage.googleapis.com/cvdf-datasets/mnist
for f in train-images-idx3-ubyte train-labels-idx1-ubyte \
         t10k-images-idx3-ubyte  t10k-labels-idx1-ubyte; do
  curl -O "$base/$f.gz" && gunzip "$f.gz"
done
# main.cu expects a dot before "idx", not a hyphen:
mv train-images-idx3-ubyte train-images.idx3-ubyte
mv train-labels-idx1-ubyte train-labels.idx1-ubyte
mv t10k-images-idx3-ubyte  t10k-images.idx3-ubyte
mv t10k-labels-idx1-ubyte  t10k-labels.idx1-ubyte

make            # builds ./ann for sm_53 (Jetson Nano) by default
./ann           # prints starting accuracy, then per-epoch test accuracy
```

Override the target GPU with `make ARCH=sm_75` (Colab/Kaggle T4), `sm_86`
(RTX 30xx), etc. Epoch count is hard-coded to 1 in `main.cu` — raise it to train
longer.

## Profiling

```bash
make profile && ./ann && gprof ann gmon.out | less   # host (CPU) side only
```

`gprof` covers host code only. For the GPU kernels, use NVIDIA's tools — all
preinstalled on Colab's CUDA image (free T4 runtime):

```bash
make ARCH=sm_75
nvprof ./ann                         # per-kernel times (deprecated but works on T4)
nsys profile -o prof ./ann           # timeline → open prof.nsys-rep in Nsight Systems
ncu --kernel-name matrix_dot_kernel ./ann   # per-kernel metrics
```

Add `-lineinfo` to `NVCCFLAGS` for source-line correlation in Nsight reports.

## Running without a local GPU

These are real CUDA kernels — they need an NVIDIA GPU at runtime. There is no
practical CPU emulation for modern CUDA (NVIDIA dropped device emulation after
CUDA 3.1). Use a free cloud GPU instead: **Google Colab** or **Kaggle**
(`Runtime → GPU`, then `nvcc … -arch=sm_75 && ./ann`), or any GPU cloud VM.

## License

MIT — see [LICENSE](LICENSE).
