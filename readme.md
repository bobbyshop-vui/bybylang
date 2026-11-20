# BybyLang

A Nim-based DSL that compiles to Nim source and then a native binary. Its
`gpu ...` commands generate tensor operations and `session:` batches supported
operations. The `micro kernel` block defines VM bytecode for Metal and Vulkan
lowering or the CPU interpreter. The DSL also supports general control flow and
hardware-simulation commands.

## Build & run

```bash
make -C vendor/bybylang build
make -C vendor/bybylang test
# Or, from vendor/bybylang itself:
make build
make test
# equivalent to the build and AOT run:
nim c -d:release -o:bybylang bybylang.nim
./bybylang demo/demo_gpu.bybylang --aot=demo/demo_gpu_out
./demo/demo_gpu_out
```

`--aot=<path>` generates `<path>.nim`, compiles it with Nim, and produces the binary `<path>`.

### Diagnostics: errors, `warn`, `error`, `traceback`

**Compile errors point at your `.bybylang` source**, in the style of clang. The compiler tags every generated Nim line with the source line it came from and translates Nim's messages back:

```text
prog.bybylang:1:11: error: expected: ')', but got: 'echo'
x = (1 + 2
          ^
1 error(s) generated.
```

- The exit status is `1`, and there is no Nim stack trace. Pass `-v` to also see the compiler log and the stack trace, `--raw-nim` to print Nim's own output as well, `--nim-warnings` to report Nim warnings of the generated code, and `-Werror` to treat warnings as errors.
- Messages like `expected ')'` are found by Nim at the token after the mistake. The caret is placed at the end of the previous statement, where the mistake is.
- A missing `import` is reported on the `import` line.
- The compiler log (`[INFO]`, `[DEBUG]`) is off by default. `-v` turns on info and `-vv` debug.

**Reporting from a program.** Three statements work in BybyLang code and report the `.bybylang` file and line (they go to stderr):

```text
warn "buffer is larger than the tile"     # prog.bybylang:12: warning: buffer is ...   (keeps running)
traceback "unexpected state"              # prints the call stack and the note        (keeps running)
error "model.fir is truncated"            # prog.bybylang:14: error: ...  then exits with status 1
```

`traceback` lists the BybyLang functions that were active, oldest first, then the line of the `traceback` statement. Using it makes the compiler build the program with Nim stack frames (`--stackTrace:on`), so only programs that use it pay for them.

**Reporting from a Nim library.** `import diag` (the module sits next to `bybylang.nim`) and call the same three words. They report the Nim file and line of the call:

```nim
import diag

proc loadTable(path: string) =
  if not fileExists(path):
    warn "table missing, using the default"     # tables.nim:5:  warning: ...
  if corrupted:
    traceback "corrupted table"                 # prints the stack, continues
    error "cannot continue"                     # prints, then raises BybyError
```

| Symbol | Meaning |
| --- | --- |
| `warn(msg)`, `diagWarn(msg, file, line)` | Print a warning. |
| `error(msg)`, `diagError(msg, file, line)` | Print an error and raise `BybyError` (an uncaught `BybyError` exits with status 1 and prints nothing more). |
| `traceback(msg = "")`, `diagTraceback(msg, file, line)` | Print the call stack; the program continues. Frames need `--stackTrace:on`. |
| `diagEmit(severity, file, line, col, msg, source, caretLen, notes)` | Print one clang-style diagnostic, with the source line and a caret when `source` and `col` are given. |
| `gDiagWarnings`, `gDiagErrors`, `gDiagWarningsAsErrors` | Counters and the `-Werror` switch. |

Colors are used only on a terminal and are off when `NO_COLOR` is set.

---

---

## 1. General syntax

- One command per line in a `.bybylang` file. Indentation (spaces/tabs) determines the block for `if/elif/else/while/for`.
- `#` at the start of a line is a comment.
- `import name` or `import "path/name.bybylang"` — recursive import, `.bybylang` extension auto-appended, cycle-protected via absolute path.
- `print <expr>` → generates `echo <expr>`.
- `function NAME` opens a definition; the body is whatever's indented deeper than the `function` line (same dedent rule as `if/while/for`), no closing marker needed. Invoke it with `call NAME` or just a line containing only `NAME`.
- `function NAME(a: int, b = 2.0): T {.pragma.}` declares parameters, a return type and pragmas exactly as a Nim `proc` header; assign `result` or use `return`. Functions are emitted in source order before the top-level statements, so a function can call those defined above it but cannot read top-level variables: pass them as parameters. Top-level `type` sections and `const` lines are emitted before the functions.
- The first plain `name = expr` declares `var name`; later ones assign. Field, index and compound assignments (`a.b = x`, `a[i] = x`, `x += 1`) and `var`/`let`/`const` lines are passed to Nim unchanged.
- Besides `if/elif/else/while/for`, the Nim block statements `case`/`of`, `try`/`except`/`finally`, `when`, `block` and `defer` nest by indentation; a `type` section is copied to Nim verbatim. Blank lines and comments never end a block.
- `if cond:` / `elif cond:` / `else:` / `while cond:` / `for x in range(a, b):` — translated directly to the equivalent Nim construct, arbitrarily nestable.
- `mode is N` (N = 1..4) → prints "Mode 1: Low-level" / "Mode 2: Mid-level" / "Mode 3: High-level" / "Mode 4: Web-level" (any other N → "Unknown mode").

## 2. `gpu ...` commands

### Select backend

```
gpu backend is "auto"      # auto | cpu | cuda (Vulkan alias) | metal | vulkan
```

`auto` probes Metal on macOS, then Vulkan, then CPU. `cuda` is a legacy backend name that maps to Vulkan; CUDA/cuBLAS is not implemented in this version. `opencl` is not a current `GpuBackend` option.

`gForbidCpuFallback` defaults to `false`. Call `setForbidCpuFallback(true)` when you need GPU-only execution to fail fast instead of allowing operation-level CPU fallback. Backend availability and operation coverage vary; check boolean results from session encoders and surface errors from `sessionEnd`.

### Declare an array

```
gpu array A = [1, 2, 3, 4]
```

Generates `var A: seq[float32] = @[1, 2, 3, 4].mapIt(it.float32)`. **Must be declared before use** — codegen has no hoisting; it translates lines strictly in file order.

### Basic binary ops

```
gpu add A, B -> C
gpu sub A, B -> C
gpu mul A, B -> C
gpu div A, B -> C
```

A trailing `size N` is descriptive only and doesn't affect codegen.

### Matmul

```
gpu matmul A, B -> C m M k K n N
```
A: [M×K], B: [K×N] → C: [M×N].

`gpuMatmul2` (two independent matmuls in one backend call) is available to Nim callers, but has no `.bybylang` syntax. Call `gpuMatmul2(...)` directly from Nim.

### Activations

```
gpu relu X -> Y
gpu sigmoid X -> Y
gpu tanh X -> Y
gpu apflu X -> Y alpha A beta B      # alpha/beta optional; defaults to alpha=0.1, beta=1.0 if omitted
```

Both `apflu_backward` and `fused_add_act` have `.bybylang` syntax and Nim APIs (`gpuApfluBackward`, `gpuFusedAddAct`).

### Softmax / LayerNorm

```
gpu softmax X -> Y rows R cols C
gpu layernorm X, GAMMA, BETA -> Y rows R cols C eps E
gpu layernorm_backward DY, X, GAMMA, BETA -> DX, DGAMMA, DBETA rows R cols C eps E
```

### Embedding

```
gpu embedding TABLE, INDICES -> Y vocab V dim D
```
`INDICES` is auto-cast to `int32`.

### Attention (fused)

```
gpu attention Q, K, V -> O, S_MATRIX B H S D scale SCALE
gpu attention_backward Q, K, V, S_MATRIX, DY -> DQ, DK, DV B H S D scale SCALE
```
B=batch, H=heads, S=seq_len, D=head_dim.

### RoPE

```
gpu rope X, COS, SIN -> Y seqlen T heads H dim D
```
Rotates `X` in place using `COS`/`SIN` tables. Use this inside `session:`; there is no immediate `gpu rope` implementation. If the backend cannot begin a session, generated code warns and returns the original input unchanged, so do not treat that fallback as a RoPE result.

### Other DSL operations

The parser also supports `sqrt`, `exp`, `log`, `pow`, `gather`, `compare`, `fused_add_act`, `apflu_backward`, `permute`, `split_slice`, `concat`, `argmax_index`, and `dropout_mask`. Their exact operand and shape syntax is demonstrated in [`demo/demo_gpu.bybylang`](demo/demo_gpu.bybylang). `compare` returns a boolean mask; gather and embedding indices are cast to `int32`.

### Generic op (fallback)

```
gpu <other_op> A, B -> C
```
Generates `gpuOp("<other_op>", backend, A, B)` — requires a matching branch inside `gpuOp` to actually do something.

---

## 3. `session:` blocks — batching multiple `gpu ...` calls

```
session:
  gpu add A, B -> C size 4
  gpu mul C, A -> D size 4
  gpu relu D -> E
  gpu rope Q, COS, SIN -> Q2 seqlen 4 heads 1 dim 4
```

Wraps the enclosed `gpu ...` lines with the Session API (`sessionBegin` / `sessionUpload` / `session<Op>` / `sessionEnd` / `sessionRead` / `sessionFree` in `gpubackend.nim`) so intermediate results stay on the GPU across ops instead of round-tripping through `seq[float32]` after every single call.

- Ops with a session-encoded path include `add`/`sub`/`mul`/`div`, activations, `softmax`, `matmul`, `layernorm`, `rope`, `fused_add_act`, `apflu`/`apflu_backward`, `gather`, `compare`, `dropout_mask`, and `split_slice`. Other operations such as `embedding`, `attention`, `attention_backward`, `layernorm_backward`, `permute`, `concat`, and `argmax_index` run through their immediate APIs when placed in the block; those calls do not become resident session ops.
- Unlike outside a session, a trailing `size N` **does** affect codegen here — it's used as the element-count argument passed to the session op, not just descriptive.
- If `sessionBegin()` fails, the generated block falls back to immediate calls for supported fallback operations. `gpu rope` has no immediate equivalent; fallback warns and leaves its output equal to its input (RoPE is skipped).
- Non-`gpu` lines inside a `session:` block (assignments, `print`, etc.) run normally, outside the session's GPU batching.

---

## 4. Nim GPU and Session APIs

Import the backend directly from a Nim application. With the project root as the working directory:

```nim
import vendor/bybylang/gpubackend

let backend = parseBackend("vulkan") # also "metal", "auto", "cpu"
let a = @[1'f32, 2, 3, 4]
let b = @[10'f32, 20, 30, 40]
let c = gpuOp("add", backend, a, b)
echo c
```

For the DSL compiler facade and VM helpers, import `vendor/bybylang/bybylang`; it re-exports `gpubackend`, `micro_kernel`, and the Metal/Vulkan modules. For backend-only code, `vendor/bybylang/gpubackend` keeps the import surface smaller.

The immediate `gpu*` API accepts/returns CPU `seq`s, so each call may transfer data. Available families include `gpuMatmul`, `gpuMatmul2`, `gpuMatmulQ4`, vector ops (`gpuRelu`, `gpuSigmoid`, `gpuTanh`, `gpuFusedAddAct`), `gpuSoftmax`, `gpuLayernorm` and backward, fused attention and backward, embedding lookup, RoPE-related session operations, and utility ops (`gpuGather`, `gpuCompare`, `gpuPermute`, `gpuSplitSlice`, `gpuConcat`, `gpuArgmaxIndex`, `gpuDropoutMask`). See the exported signatures in `gpubackend.nim` for exact dimensions and argument order.

For a chain of GPU operations, use `SessionHandle` so intermediate tensors stay on the device. Always check encoder return values and end the session before reading results:

```nim
import vendor/bybylang/gpubackend

let backend = parseBackend("vulkan")
if not sessionBegin(backend):
  raise newException(IOError, "GPU sessions are unavailable")
var a = sessionUpload(backend, @[1'f32, 2, 3, 4])
var b = sessionUpload(backend, @[10'f32, 20, 30, 40])
var c: SessionHandle
if not sessionVecOp(backend, "add", a, b, c, 4):
  raise newException(IOError, "add encode failed")
if not sessionEnd(backend):
  raise newException(IOError, sessionGetLastError(backend))
echo sessionRead(backend, c, 4)
sessionFree(backend, a)
sessionFree(backend, b)
sessionFree(backend, c)
```

Core lifecycle calls: `sessionBegin`, `sessionUpload`/`sessionUploadBorrowed`/`sessionUploadPrivate`, `sessionAllocScratch`, `sessionEnd`, `sessionRead`, and `sessionFree`. Encode APIs cover vector/activation ops, matmul and Q4 matmul, softmax, layernorm/RMSNorm, RoPE, attention, copy/scale, gather/compare, dropout, slicing, and embedding where implemented by the chosen backend. `sessionCommitContinue` can commit queued work and continue encoding without an explicit CPU readback; `sessionRecoverQueue` is for recovery after a failed session. `sessionAllocHeap` is currently a Metal-only heap optimization and returns `nil` on unsupported backends; check it before suballocating.

Prefill and long-running work:

| Call | What it does |
| --- | --- |
| `sessionFlashAttentionPrepare(backend, D): string` | Picks the prefill flash-attention kernel for head dim `D` (call outside a session). Returns `"coop"` (Vulkan tensor cores), `"scalar"` (Metal, Vulkan without tensor cores), or `""` when there is none. Vulkan checks its kernels against a reference first and drops one that fails. |
| `sessionFlashAttentionPrefillEnc(backend, q, kAll, vAll, o, T, H, HKV, D, pos, maxSeqLen, scale): bool` | Causal flash attention for `T` new rows. `q`/`o` are `[T, H, D]`; `kAll`/`vAll` are the cache `[HKV, maxSeqLen, D]` already holding positions `0 .. pos+T-1`; GQA maps head `h` to `h div (H div HKV)`. |
| `sessionKvScatterEnc(backend, src, dst, T, HKV, D, pos, maxSeqLen): bool` | KV cache append: `dst[h, pos+t, :] = src[t, h, :]`. |
| `sessionPacesCommandBuffers(backend): bool` | True when long GPU work should be split with `sessionCommitContinue` (Metal on Intel/AMD Macs, where the watchdog kills a command buffer that runs for seconds; Vulkan). False on Apple Silicon. |

`sessionMatmulQ4` with `M > 1` takes the tiled prefill GEMM: on Metal a
64x128 register-tiled kernel (`BYBY_METAL_Q4_GEMM=0` turns it off,
`BYBY_METAL_Q4_GEMM_MIN_M` sets the smallest M, default 8; needs `K % 32 == 0`
and a group size that is 0 or a multiple of 32); on a Vulkan GPU with
`KHR_cooperative_matrix` a tensor-core GEMM split-K over every SM/CU.

For quantized Q4 weights, upload once and reuse the returned resident handle across sessions:

```nim
var q4 = residentUploadQ4(backend, packedWeights, scales, zeros,
                          k = 4096, n = 4096, groupSize = 32,
                          nGroupsPerRow = 128)
discard residentPrepareQ4(q4, M = 1) # optional autotune; false means no tuned plan
if not sessionBegin(backend): raise newException(IOError, "session unavailable")
var x = sessionUpload(backend, activations)
var y: SessionHandle
if not sessionMatmulQ4(backend, x, q4, y, M = 1):
  raise newException(IOError, "Q4 matmul encode failed")
if not sessionEnd(backend): raise newException(IOError, sessionGetLastError(backend))
let result = sessionRead(backend, y, 4096)
sessionFree(backend, x)
sessionFree(backend, y)
residentFreeQ4(q4) # call only when unloading the weights
```

The old `cu*` resident-tensor names are compatibility stubs in this revision; they do not provide CUDA/cuBLAS execution. Use the common Session API above. The legacy backend string `cuda` resolves to Vulkan.

### Command replay

Use `sessionCanRecord(backend)` before choosing the replay path. Allocate all
persistent inputs, outputs and scratch first. Call `sessionBeginRecording`,
encode the fixed-shape step and verify `sessionEnd`. `sessionLastRecording`
returns a recording ID or `-1` if the runtime could not retain the commands.
Staging uploads, deferred frees and internal scratch can prevent retention.

Bindings and launch sizes remain fixed. Change inputs with `sessionHostWrite`
on mapped buffers allocated by `sessionHostBuffer`. Encode a copy to the
mapped output and call `sessionHostReadBarrier` in the recorded step if the
host reads results. `sessionReplay` submits and waits outside an active
session. Call `sessionFreeRecording` before freeing the referenced buffers.
The lifecycle and expected results are exercised by the Metal replay test.
[Replay test](tests/t_metal_session_replay.nim)

Vulkan replay reuses a command buffer with its descriptor pools. Metal can
use an indirect command buffer or a plain encoder. `sessionRecordingNote`
reports the Metal path or why the recording was refused, and
`BYBY_METAL_REPLAY_ICB=0` selects the plain path.

### Tile GEMM and SSD prefill

- `sessionFp4GemmTensorCore`, `sessionFp4GemmTiledEnc`, and
  `sessionFp4GemmMetal` compute `C[M, N] = A[M, K] @ W[N, K].T` from FP4-KM
  weights. The packed weight slot holds nibble bytes followed by one scale
  byte per 32 weights. The LUT slot holds 16 FP4 values and 256 scale values as
  `float32`. The first Vulkan encoder requests cooperative matrices, the
  second uses register tiles. Metal uses register tiles or Apple
  `simdgroup_matrix`, with a caller-allocated output handle.
- `sessionF16GemmTile` uses IEEE FP16 weights in `[N, K]` order on Vulkan.
  Select cooperative matrices with `tensorCore = true` or register tiles with
  `false`. FP16 and FP4-KM need whole K slices. Vulkan cooperative tiles need
  `K` divisible by 64 and `N >= 16`. Register tiles need `K` divisible by 32.
- `sessionTileGemmReady(backend, f16, tensorCore)` checks a Vulkan tile path
  before selecting it. The first call for a path opens a small parity-check
  session of its own. A GEMM encoder returning `false` lets the caller choose
  a fallback.
- `sessionMamba2SsdPrepare(backend, N, D, Q)` picks and checks a Vulkan kernel
  for the state size, head dimension and chunk length. Call it outside an
  active session. It returns `"coop"`, `"scalar"`, or `""` when unavailable.
  `sessionMamba2SsdPrefillEnc` applies the chunked pass with state carry in a
  caller-supplied `float32` arena.
- `sessionCompileSeconds()` accumulates shader and pipeline creation time on
  cache misses. Sample it before and after a phase to measure JIT setup.

`buildFp4TileGemmProgram` and `buildF16TileGemmProgram` produce tile VM
programs. The cooperative Vulkan lowering fetches the next K slice into
registers while the current slice is processed. The tests compare tile
results against host references and compile unused GPU routes where tools are
available. [Tile lowering](micro_kernel/tile_gemm.nim) ·
[Tile regression test](tests/t_fp4_tile_gemm.nim)

## 5. Native BybyLang micro-kernels

The `.bybylang` compiler builds a `VmProgram` from the `micro kernel` block.
The CPU VM interprets it, while GPU execution lowers it to Metal or Vulkan
and reads the output after dispatch.

```bybylang
gpu backend is "cpu"
gpu array INPUT = [1, 2, 3, 4, 5, 6, 7, 8]

micro kernel scale INPUT -> OUTPUT threads 8 tg 4 out 8:
  emitUTgid 0
  emitUMulK 0 0 4
  emitULid 1
  emitUAdd 0 0 1
  emitULoadIn 1 0 0
  emitLoadConstF32 2 2.0'f32
  emitMul 1 1 2
  emitUStoreF32 1 0 0
  emitHalt

print OUTPUT
```

The VM instruction API is exported by `vendor/bybylang/micro_kernel` and re-exported by `vendor/bybylang/bybylang`: create a `VmProgram` and append instructions with `emit*` procedures. Instructions include float/integer loads and stores, arithmetic, FMA, loops and conditionals, thread/threadgroup IDs, threadgroup memory and barriers, atomics, and virtual-threadgroup helpers. Integer registers are 32-bit; float registers are `float32`. Each instruction takes register indices and offsets as documented at its declaration in `micro_kernel/session_vm.nim`.

The header sets total dispatch threads, threads per group and output elements.
The example uses `group_id * 4 + local_id` to cover both groups and prints
`@[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]`. For a custom dispatch,
choose counts or guard accesses so each thread stays in bounds. Nim callers
can build the `VmProgram` and call
`executeMicroKernel(backend, kernelName, program, input, outputElems, threads, tgSize)`
directly.

### Control flow and scopes

`emitForBegin`/`emitForEnd`, `emitIfBegin`/`emitIfEnd` and `emitVtgBegin`/`emitVtgEnd` open and close scopes, and **scopes nest in any combination**: an `if` inside an `if`, an `if` inside a `for`, a `for` inside a `vtg`, and so on. The condition of `emitIfBegin cond` is `R[cond] != 0.0` (build it with `emitCmpLt`). Rules:

- Every `*Begin` needs its matching `*End`, closed innermost first.
- Keep barriers (`emitBarrierTg`, `emitGridBarrier`) outside `if` bodies, because every thread of the threadgroup must reach them.
- A `for` count must be uniform across the threadgroup when a barrier is inside it.

```text
micro kernel pick INPUT -> OUTPUT threads 8 tg 8 out 8:
  emitULid 0
  emitLoadConstF32 1 2.0'f32
  emitLoadConstF32 2 4.0'f32
  emitCmpLt 3 0 1                 # R3 = (lid < 2)
  emitIfBegin 3
    emitCmpLt 4 0 2               # R4 = (lid < 4)
    emitIfBegin 4                 # nested if
      emitLoadConstF32 5 3.0'f32
    emitIfEnd
  emitIfEnd
  emitUStoreF32 5 0 0
  emitHalt
```

Nested scopes run on Metal, Vulkan and the CPU backend.

### Wide input load

`emitULoadIn4 rd ub off` loads `R[rd..rd+3]` from four consecutive input floats at `U[ub]+off` in one 16-byte access (`rd + 3 <= 31`). The address must be a multiple of 4 floats.

### Grid control buffer

`emitGridBarrier ctrl` and `emitGridEpilogue ctrl` take the float index of a four-word control block (`cnt, done, err, lastErr`) inside the memory buffer. Passing the sentinel `0xFFFFFFFF` as `ctrl` instead puts the control block in its own buffer at Metal slot 5 (`ctrl_`). A kernel that uses the sentinel reads data written by other threadgroups through relaxed atomic loads (`ldShared`), so it sees them within the same dispatch. GLSL (Vulkan) does not support the grid barrier yet.

### Virtual threadgroups

`emitVtgBegin mode nVirt param` runs `nVirt` virtual threadgroups on the physical grid. Mode `0` is the plain form, mode `2` is the guard form. Mode `1` (sub-threadgroups) packs several virtual threadgroups into one physical threadgroup, and its `param` is `(S << 20) | stride`:

- `S` is the lanes per virtual threadgroup (`0` means 32). It must be a power of two that divides the physical threadgroup size.
- `stride` (low 20 bits) is the threadgroup-memory floats of each virtual threadgroup.

`param` values below `1 << 20` keep their old meaning (`S = 32`, `stride = param`).

### CPU backend

The CPU VM keeps byte (`u8`), look-up table, input and float memory in separate arrays (`nimf_cpu_vm_run_slots`), so a program sees the same slot layout as on the GPU. `nimf_cpu_vm_run` is kept and treats one float arena as every slot. An out-of-range index reports an error instead of reading another slot.

### Core recording for FIR

Run with `BYBY_FIR_PROBE=run.fprobe` to record backend calls, buffer metadata
and registered VM words. Programs already registered by the compiler remain
available when recording starts. `BYBY_FIR_PROBE_STEPS=decode:2,prefill:1`
limits explicitly marked steps. `BYBY_FIR_PROBE_DATA=1` opts in to exact upload
bytes for general file recordings. Drivers may otherwise record short values
and fingerprints. `firProbeStart` and `firProbeStop` control the recorder from Nim.

A parent process can provide `BYBY_FIR_PROBE=fd:T[,D]`, passing inherited file
descriptors for trace records and optional upload data. The recorder sends the
trace when it stops. A trace descriptor alone supports fingerprints without
an upload payload stream. [Recorder implementation](probe.nim)

---

## 6. Low-level hardware-simulation commands

These **do** have real DSL syntax, parsed in `genBlock` (`bybylang.nim` ~line 668–714) and lowered to calls that operate on a simulated RAM (1024 ints) / BUS (seq[string]) / 32 Pins:

```
apu tran "chip1" with 101010          # -> apuTran("chip1", 101010)
apu mem write RAM0 with 42            # -> apuMem("write", "RAM0", "42")
apu mem read RAM0 with 0              # -> apuMem("read", "RAM0", "0")
apu core run                          # -> apuCore(1, "run")  (mode is always hardcoded to 1)
apu pin 3 is high                     # -> apuPin(3, "high")
bit send 1010                         # -> bitSend("1010")
bit recv                              # -> bitRecv()
mem map "device0"                     # -> memMap("device0")
mem push RAM0 with 99                 # -> memPush("RAM0", "99")
tran pulse pin 3 width 10ns           # -> tranPulse(3, "10ns")
```

Parser quirks worth knowing:
- `apu mem <action> <target> with <value>` — `target` is read positionally as the 4th word (`left[3]`) and quotes are stripped; `action` should be `write` or `read`.
- `apu core ...` ignores everything after `apu core` — it always emits `apuCore(1, "run")`.
- `apu pin <n> is <state>` reads `n` from word index 2 and `state` from word index 4 — extra or missing words will misparse silently.
- `tran pulse pin <n> width <w>` reads `n` from word index 3 and `w` as the **last** word on the line.

`apuTran`, `apuMem`, `apuCore`, `apuPin`, `bitSend`, `bitRecv`, `memMap`, `memPush`, `tranPulse` show up as "declared but not used" warnings only when compiling `bybylang.nim` itself (the compiler for the DSL) — that warning is irrelevant to whether the DSL syntax works. The AOT-generated output (`--aot=...nim`) re-declares its own copies of these same procs (see `bybylang.nim` ~line 775 onward) so the compiled program can actually call them at runtime.

---

## 7. Backends

| Backend | File | Notes |
|---|---|---|
| Metal | `backends/metal/metal_backend.nim`, `backends/metal/kernels/` | macOS GPU; native buffer, pipeline, and session support |
| Vulkan | `backends/vulkan/vulkan_driver.nim`, `backends/vulkan/kernels/` | Vulkan GPU backend for NVIDIA, AMD, and Intel devices where drivers/features support the operation |
| CPU | reference operations in `gpubackend.nim` | fallback or explicit CPU implementation where implemented; can be forbidden with `setForbidCpuFallback(true)` |

`cuda` is accepted as a compatibility spelling and normalized to Vulkan. It does not load CUDA Runtime or cuBLAS. OpenCL is not implemented as a current selectable backend.

## 8. Minimal example

```
gpu backend is "vulkan"
gpu array A = [1, 2, 3, 4]
gpu array B = [5, 6, 7, 8]
gpu add A, B -> C
print C
```
