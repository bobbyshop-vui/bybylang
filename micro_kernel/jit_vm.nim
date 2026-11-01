## jit_vm.nim — JIT bytecode theo HotSpot + tinygrad MetalCompiler
##
## Pipeline:
##
##   Bytecode (.bvm / VmProgram)
##        │
##        ├─► Interpreter (micro_vm_kernel) + hot counter
##        │
##        └─► once hot ≥ threshold
##                 ├─► lower bytecode → Metal source (string)
##                 ├─► Metal compile (newLibraryWithSource / MTLCompiler)
##                 ├─► code cache (key = program hash + meta)
##                 └─► next time: dispatch the JIT kernel, skip the interpreter
##
## The existing kernels (matmul_q4, vecop, ...) are UNCHANGED — the production hot path.
## JIT is only for micro-VM programs / custom logic.

import std/[tables, strutils, strformat, times, os, pegs, hashes]
import ./session_vm
when defined(macosx):
  import ./session_vm_dispatch
  # the GPU backend lives in vendor/bybylang/gpubackend (same bybylang package)
  import ../gpubackend as byby

# ── Tier configuration (HotSpot-style) ────────────────────────────

type
  JitTier* = enum
    tierInterpret = 0  ## micro_vm only
    tierProfile   = 1  ## interpret + count
    tierCompiled  = 2  ## has a Metal pipeline from the JIT

  JitConfig* = object
    hotThreshold*: int     ## runs before JIT (default 10)
    enableMetalJit*: bool  ## compile ra Metal source
    enableVulkanJit*: bool ## compile ra GLSL source (Vulkan)
    enableCpuJit*: bool    ## (reserved) LLVM ORC — not wired yet
    maxCacheEntries*: int

  CompiledEntry* = object
    tier*: JitTier
    hitCount*: int
    metalSource*: string   ## lowered source (debug / recompile)
    glslSource*: string    ## lowered GLSL source (Vulkan path)
    kernelName*: string    ## name in the library after JIT
    compiledAt*: float
    ## the real pipeline handle lives on the C/Metal side; Nim keeps the name + key

  JitCache* = object
    cfg*: JitConfig
    entries*: Table[string, CompiledEntry]  ## key = programKey

proc defaultJitConfig*(): JitConfig =
  JitConfig(
    hotThreshold: 10,
    enableMetalJit: true,
    enableVulkanJit: true,
    enableCpuJit: false,
    maxCacheEntries: 64
  )

proc initJitCache*(cfg = defaultJitConfig()): JitCache =
  JitCache(cfg: cfg, entries: initTable[string, CompiledEntry]())

# ── Lower bytecode → Metal source ────────────────────────────
## NO for/while loop on the host: the decoder runs by tail recursion.
## (Generating Metal TEXT has to happen on the host - the GPU cannot write its own source -
## but that is string processing, not arithmetic. All arithmetic (hashing...)
## moved to a micro-kernel, see programKey.)
## The existing kernels are generated EXACTLY as before (diffed against the original).

type LowerState = object
  body: seq[string]
  pc, loopDepth, indent: int
  usesU, usesIn: bool
  glsl: bool
  od: bool              ## uses one-dispatch ops (persistent kernel) -> the kernel has its own prologue
  vtgModes: seq[uint32] ## mode stack of the open OP_VTG_BEGIN blocks

proc ind(st: LowerState): string = "  ".repeat(st.indent)

proc imm(st: var LowerState; p: VmProgram): uint32 =
  ## Reads the next operand word (0 past the end of the program) and skips it.
  if st.pc < p.words.len:
    result = p.words[st.pc]
    inc st.pc

proc lowerStep(st: var LowerState; p: VmProgram) =
  let w = p.words[st.pc]
  inc st.pc
  let op = w and 0xFF'u32
  let a = (w shr 8) and 0x1F
  let b = (w shr 13) and 0x1F
  let c = (w shr 18) and 0x1F
  if op >= OP_U_TGID and op <= OP_U_ATOMIC_ADD: st.usesU = true
  if op == OP_U_LOAD_IN or op == OP_U_LOAD_IN4: st.usesIn = true
  case op
  of OP_NOP:
    discard
  of OP_HALT:
    st.body.add st.ind() & "return;"
  of OP_LOAD_F32:
    let off = st.imm(p)
    if st.glsl:
      st.body.add st.ind() & &"R[{a}] = memF32[uint(R[{b}]) + {off}u];"
    else:
      st.body.add st.ind() & &"R[{a}] = memF32[(uint)R[{b}] + {off}u];"
  of OP_STORE_F32:
    let off = st.imm(p)
    if st.glsl:
      st.body.add st.ind() & &"memF32[uint(R[{b}]) + {off}u] = R[{a}];"
    else:
      st.body.add st.ind() & &"memF32[(uint)R[{b}] + {off}u] = R[{a}];"
  of OP_LOAD_U8:
    let off = st.imm(p)
    if st.glsl:
      let idx = &"(uint(R[{b}]) + {off}u)"
      st.body.add st.ind() & &"R[{a}] = float((memU8[{idx} >> 2u] >> (({idx} & 3u) * 8u)) & 255u);"
    else:
      st.body.add st.ind() & &"R[{a}] = (float)memU8[(uint)R[{b}] + {off}u];"
  of OP_LOAD_CONST:
    let bits = st.imm(p)
    if st.glsl:
      st.body.add st.ind() & &"R[{a}] = uintBitsToFloat({bits}u);"
    else:
      st.body.add st.ind() & &"R[{a}] = as_type<float>({bits}u);"
  of OP_ADD:
    st.body.add st.ind() & &"R[{a}] = R[{b}] + R[{c}];"
  of OP_MUL:
    st.body.add st.ind() & &"R[{a}] = R[{b}] * R[{c}];"
  of OP_FMA:
    let rc = st.imm(p) and 0x1F
    st.body.add st.ind() & &"R[{a}] = fma(R[{b}], R[{c}], R[{rc}]);"
  of OP_LUT_U8:
    if st.glsl:
      st.body.add st.ind() & &"R[{a}] = lut[uint(R[{b}]) & 255u];"
    else:
      st.body.add st.ind() & &"R[{a}] = lut[((uint)R[{b}]) & 255u];"
  of OP_SUB:
    st.body.add st.ind() & &"R[{a}] = R[{b}] - R[{c}];"
  of OP_DIV:
    st.body.add st.ind() & &"R[{a}] = R[{b}] / R[{c}];"
  of OP_EXP:
    st.body.add st.ind() & &"R[{a}] = exp(R[{b}]);"
  of OP_NEG:
    st.body.add st.ind() & &"R[{a}] = -R[{b}];"
  of OP_SIGMOID:
    st.body.add st.ind() & &"R[{a}] = 1.0f/(1.0f+exp(-R[{b}]));"
  of OP_SOFTPLUS:
    st.body.add st.ind() & &"R[{a}] = log(1.0f+exp(R[{b}]));"
  of OP_RSQRT:
    st.body.add st.ind() & &"R[{a}] = rsqrt(R[{b}]);"
  of OP_LOAD_F32_IDX:
    if st.glsl:
      st.body.add st.ind() & &"R[{a}] = memF32[uint(R[{b}])];"
    else:
      st.body.add st.ind() & &"R[{a}] = memF32[(uint)R[{b}]];"
  of OP_LOAD_LID:
    if st.glsl:
      st.body.add st.ind() & &"R[{a}] = float(lid);"
    else:
      st.body.add st.ind() & &"R[{a}] = (float)lid;"
  of OP_LOAD_TGID:
    if st.glsl:
      st.body.add st.ind() & &"R[{a}] = float(tgid);"
    else:
      st.body.add st.ind() & &"R[{a}] = (float)tgid;"
  of OP_STORE_F32_IDX:
    if st.glsl:
      st.body.add st.ind() & &"memF32[uint(R[{b}])] = R[{a}];"
    else:
      st.body.add st.ind() & &"memF32[(uint)R[{b}]] = R[{a}];"
  of OP_CMP_LT:
    st.body.add st.ind() & &"R[{a}] = (R[{b}] < R[{c}]) ? 1.0f : 0.0f;"
  of OP_IF_BEGIN:
    st.body.add st.ind() & &"if (R[{a}] != 0.0f) {{"
    inc st.loopDepth   # shares the bracket counter with FOR (enough: we only need to know whether a block is still open at the end)
    inc st.indent
  of OP_IF_END:
    if st.loopDepth > 0:
      dec st.indent
      dec st.loopDepth
      st.body.add st.ind() & "}"
  of OP_ADD_CONST:
    let bits = st.imm(p)
    if st.glsl:
      st.body.add st.ind() & &"R[{a}] = R[{b}] + uintBitsToFloat({bits}u);"
    else:
      st.body.add st.ind() & &"R[{a}] = R[{b}] + as_type<float>({bits}u);"
  of OP_MUL_CONST:
    let bits = st.imm(p)
    if st.glsl:
      st.body.add st.ind() & &"R[{a}] = R[{b}] * uintBitsToFloat({bits}u);"
    else:
      st.body.add st.ind() & &"R[{a}] = R[{b}] * as_type<float>({bits}u);"
  of OP_FOR_BEGIN:
    if st.glsl:
      st.body.add st.ind() & &"for (int _l{st.loopDepth}=0; _l{st.loopDepth}<int(R[{a}]); ++_l{st.loopDepth}) {{"
    else:
      st.body.add st.ind() & &"for (int _l{st.loopDepth}=0; _l{st.loopDepth}<(int)R[{a}]; ++_l{st.loopDepth}) {{"
    inc st.loopDepth
    inc st.indent
  of OP_FOR_END:
    if st.loopDepth > 0:
      dec st.indent
      dec st.loopDepth
      st.body.add st.ind() & "}"
  of OP_BARRIER_TG:
    st.body.add st.ind() & (if st.glsl: "memoryBarrierShared(); barrier();"
                             else: "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);")
  of OP_TG_LOAD_F32:
    let off = st.imm(p)
    if st.glsl:
      st.body.add st.ind() & &"R[{a}] = tgF32[uint(R[{b}]) + {off}u];"
    else:
      st.body.add st.ind() & &"R[{a}] = tgF32[(uint)R[{b}] + {off}u];"
  of OP_TG_STORE_F32:
    let off = st.imm(p)
    if st.glsl:
      st.body.add st.ind() & &"tgF32[uint(R[{b}]) + {off}u] = R[{a}];"
    else:
      st.body.add st.ind() & &"tgF32[(uint)R[{b}] + {off}u] = R[{a}];"
  of OP_U_TGID:
    st.body.add st.ind() & &"U[{a}] = tgid;"
  of OP_U_LID:
    st.body.add st.ind() & &"U[{a}] = lid;"
  of OP_U_CONST:
    let k = st.imm(p)
    st.body.add st.ind() & &"U[{a}] = {k}u;"
  of OP_U_ADD:
    st.body.add st.ind() & &"U[{a}] = U[{b}] + U[{c}];"
  of OP_U_MUL:
    st.body.add st.ind() & &"U[{a}] = U[{b}] * U[{c}];"
  of OP_U_SHR:
    st.body.add st.ind() & &"U[{a}] = U[{b}] >> U[{c}];"
  of OP_U_ADDK:
    let k = st.imm(p)
    st.body.add st.ind() & &"U[{a}] = U[{b}] + {k}u;"
  of OP_U_MULK:
    let k = st.imm(p)
    st.body.add st.ind() & &"U[{a}] = U[{b}] * {k}u;"
  of OP_U_ANDK:
    let k = st.imm(p)
    st.body.add st.ind() & &"U[{a}] = U[{b}] & {k}u;"
  of OP_U_SHRK:
    let k = st.imm(p)
    st.body.add st.ind() & &"U[{a}] = U[{b}] >> {k}u;"
  of OP_U_SHLK:
    let k = st.imm(p)
    st.body.add st.ind() & &"U[{a}] = U[{b}] << {k}u;"
  of OP_U_LOAD_U8:
    let k = st.imm(p)
    if st.glsl:
      let idx = &"(U[{b}] + {k}u)"
      st.body.add st.ind() & &"U[{a}] = (memU8[{idx} >> 2u] >> (({idx} & 3u) * 8u)) & 255u;"
    else:
      st.body.add st.ind() & &"U[{a}] = (uint)memU8[U[{b}] + {k}u];"
  of OP_U_LOAD_U32:
    let k = st.imm(p)
    if st.glsl:
      st.body.add st.ind() & &"U[{a}] = memU8[(U[{b}] + {k}u) >> 2u];"
    else:
      st.body.add st.ind() & &"U[{a}] = *((device const uint*)(memU8 + U[{b}] + {k}u));"
  of OP_U_LOAD_CONSTF:
    let k = st.imm(p)
    st.body.add st.ind() & &"R[{a}] = lut[U[{b}] + {k}u];"
  of OP_U_LOAD_IN:
    let k = st.imm(p)
    st.body.add st.ind() & &"R[{a}] = inF32[U[{b}] + {k}u];"
  of OP_U_LOAD_IN4:
    let k = st.imm(p)
    if st.glsl:
      st.body.add st.ind() & &"{{ vec4 v4 = vec4(inF32[U[{b}] + {k}u], inF32[U[{b}] + {k}u + 1u], inF32[U[{b}] + {k}u + 2u], inF32[U[{b}] + {k}u + 3u]);" &
        &" R[{a}] = v4.x; R[{a}+1] = v4.y; R[{a}+2] = v4.z; R[{a}+3] = v4.w; }}"
    else:
      st.body.add st.ind() & &"{{ float4 v4 = *(device const float4*)(inF32 + U[{b}] + {k}u);" &
        &" R[{a}] = v4.x; R[{a}+1] = v4.y; R[{a}+2] = v4.z; R[{a}+3] = v4.w; }}"
  of OP_U_STORE_F32:
    let k = st.imm(p)
    st.body.add st.ind() & &"memF32[U[{b}] + {k}u] = R[{a}];"
  of OP_U_LOAD_F32:
    let k = st.imm(p)
    st.body.add st.ind() & &"R[{a}] = memF32[U[{b}] + {k}u];"
  of OP_U_ATOMIC_ADD:
    let k = st.imm(p)
    if st.glsl:
      st.body.add st.ind() & &"atomicAddFloat(U[{b}] + {k}u, R[{a}]);"
    else:
      st.body.add st.ind() & &"atomic_fetch_add_explicit((device atomic_float*)&memF32[U[{b}] + {k}u], R[{a}], memory_order_relaxed);"
  of OP_U_TO_R:
    if st.glsl:
      st.body.add st.ind() & &"R[{a}] = float(U[{b}]);"
    else:
      st.body.add st.ind() & &"R[{a}] = (float)U[{b}];"
  of OP_R_TO_U:
    if st.glsl:
      st.body.add st.ind() & &"U[{a}] = uint(R[{b}]);"
    else:
      st.body.add st.ind() & &"U[{a}] = (uint)R[{b}];"
  of OP_GRID_BARRIER:
    let ctrl = st.imm(p)
    st.od = true
    if st.glsl:
      st.body.add st.ind() & "/* OP_GRID_BARRIER is not supported in GLSL yet */ return;"
    else:
      st.body.add st.ind() & "{"
      st.body.add st.ind() & &"  device atomic_uint* gbc = (device atomic_uint*)((device uint*)memF32_ + {ctrl}u);"
      st.body.add st.ind() & "  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);"
      st.body.add st.ind() & "  gbEpoch += 1u;"
      st.body.add st.ind() & "  if (gLid == 0u) {"
      st.body.add st.ind() & "    atomic_fetch_add_explicit(gbc, 1u, memory_order_relaxed);"
      st.body.add st.ind() & "    const uint gbTarget = gbEpoch * gNtg;"
      st.body.add st.ind() & "    uint gbSpin = 0u;"
      st.body.add st.ind() & "    while (atomic_fetch_add_explicit(gbc, 0u, memory_order_relaxed) < gbTarget) {"
      st.body.add st.ind() & "      if (++gbSpin > 2000000000u) { atomic_fetch_or_explicit(gbc + 2, 1u, memory_order_relaxed); break; }"
      st.body.add st.ind() & "    }"
      st.body.add st.ind() & "  }"
      st.body.add st.ind() & "  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);"
      st.body.add st.ind() & "}"
  of OP_GRID_EPILOGUE:
    let ctrl = st.imm(p)
    st.od = true
    if not st.glsl:
      st.body.add st.ind() & "if (gLid == 0u) {"
      st.body.add st.ind() & &"  device atomic_uint* gbc = (device atomic_uint*)((device uint*)memF32_ + {ctrl}u);"
      st.body.add st.ind() & "  const uint gbDone = atomic_fetch_add_explicit(gbc + 1, 1u, memory_order_relaxed);"
      st.body.add st.ind() & "  if (gbDone + 1u == gNtg) {"
      st.body.add st.ind() & "    atomic_fetch_or_explicit(gbc + 3, atomic_fetch_add_explicit(gbc + 2, 0u, memory_order_relaxed), memory_order_relaxed);"
      st.body.add st.ind() & "    atomic_store_explicit(gbc + 2, 0u, memory_order_relaxed);"
      st.body.add st.ind() & "    atomic_store_explicit(gbc + 1, 0u, memory_order_relaxed);"
      st.body.add st.ind() & "    atomic_store_explicit(gbc + 0, 0u, memory_order_relaxed);"
      st.body.add st.ind() & "  }"
      st.body.add st.ind() & "}"
  of OP_BASE_F32:
    st.od = true
    st.body.add st.ind() & &"memF32 = memF32_ + U[{a}];"
  of OP_BASE_U8:
    st.od = true
    st.body.add st.ind() & &"memU8 = memU8_ + U[{a}];"
  of OP_BASE_LUT:
    st.od = true
    st.body.add st.ind() & &"lut = lut_ + U[{a}];"
  of OP_BASE_IN:
    st.od = true
    st.body.add st.ind() & &"inF32 = inF32_ + U[{a}];"
  of OP_U_LOAD_META:
    let k = st.imm(p)
    st.od = true
    st.body.add st.ind() & &"U[{a}] = as_type<uint>(lut_[U[{b}] + {k}u]);"
  of OP_LOAD_VACT:
    st.body.add st.ind() & &"R[{a}] = vact ? 1.0f : 0.0f;"
  of OP_VTG_BEGIN:
    let nVirt = st.imm(p)
    let prm = st.imm(p)
    st.od = true
    st.vtgModes.add a
    if a == 1:
      # V=32 < tgs: each physical TG runs tgs/32 virtual threadgroups in parallel (sub), each sub with its own tgF32 region.
      # FIXED loop count (uniform across the TG) so the barriers inside stay aligned.
      st.body.add st.ind() & "{"
      st.body.add st.ind() & &"  const uint vSubs = gTgs / 32u;"
      st.body.add st.ind() & "  const uint vSub = gLid / 32u;"
      st.body.add st.ind() & &"  const uint vTrip = ({nVirt}u + gNtg * vSubs - 1u) / (gNtg * vSubs);"
      st.body.add st.ind() & "  for (uint vIt = 0u; vIt < vTrip; ++vIt) {"
      st.body.add st.ind() & &"    const uint vRaw = (vIt * gNtg + gTgid) * vSubs + vSub;"
      st.body.add st.ind() & &"    const bool vact = vRaw < {nVirt}u;"
      st.body.add st.ind() & &"    const uint tgid = vact ? vRaw : ({nVirt}u - 1u);"
      st.body.add st.ind() & "    const uint lid = gLid & 31u;"
      st.body.add st.ind() & "    const uint tid = tgid * 32u + lid;"
      st.body.add st.ind() & &"    threadgroup float* tgF32 = tgF32_ + vSub * {prm}u;"
      st.body.add st.ind() & "    float R[32]; for (int i=0;i<32;++i) R[i]=0.f; R[0] = (float)tid;"
      st.body.add st.ind() & "    uint U[32]; for (int i=0;i<32;++i) U[i]=0u;"
      st.indent += 2
    else:
      st.body.add st.ind() & &"for (uint vIt = gTgid; vIt < {nVirt}u; vIt += gNtg) {{"
      st.body.add st.ind() & "  const uint tgid = vIt;"
      st.body.add st.ind() & "  const uint lid = gLid;"
      st.body.add st.ind() & "  const uint tid = vIt * gTgs + gLid;"
      st.body.add st.ind() & "  const bool vact = true;"
      st.body.add st.ind() & "  threadgroup float* tgF32 = tgF32_;"
      st.body.add st.ind() & "  float R[32]; for (int i=0;i<32;++i) R[i]=0.f; R[0] = (float)tid;"
      st.body.add st.ind() & "  uint U[32]; for (int i=0;i<32;++i) U[i]=0u;"
      st.indent += 1
      if a == 2:
        # mode guard: run the body only if tid < NT (the body must NOT contain barrier_tg)
        st.body.add st.ind() & &"if (tid < {prm}u) {{"
        st.indent += 1
  of OP_VTG_END:
    if st.vtgModes.len > 0:
      let m = st.vtgModes.pop()
      if m == 1:
        st.indent -= 2
        st.body.add st.ind() & "  }"   # dong for vIt
        st.body.add st.ind() & "}"     # dong khoi {
      elif m == 2:
        st.indent -= 2
        st.body.add st.ind() & "  }"   # dong if guard
        st.body.add st.ind() & "}"     # dong for
      else:
        st.indent -= 1
        st.body.add st.ind() & "}"
  else:
    st.body.add st.ind() & &"/* unknown op {op} */ return;"

proc lowerAll(st: var LowerState; p: VmProgram) =
  ## tail recursion instead of `while pc < n`
  if st.pc >= p.words.len: return
  st.lowerStep(p)
  st.lowerAll(p)

proc closeBlocks(st: var LowerState) =
  ## closes the remaining open FOR/IF blocks (instead of `while loopDepth > 0`)
  if st.loopDepth <= 0: return
  dec st.indent
  dec st.loopDepth
  st.body.add st.ind() & "}"
  st.closeBlocks()

proc lowerBytecodeToMetal*(p: VmProgram; kernelName: string): string =
  ## Generates one Metal kernel from bytecode. Every thread: tid → R0.
  var st = LowerState(indent: 2)
  st.lowerAll(p)
  st.closeBlocks()
  if st.od:
    ## ── ONE-DISPATCH persistent kernel ───────────────────────────────────
    ## The original *_ names keep the buffers; memF32/memU8/lut/inF32 are pointers whose base MAY change
    ## (OP_BASE_*). tid/lid/tgid inside a VTG block are shadowed by the VIRTUAL values.
    let hdr = @["  device float* memF32 = memF32_;",
                "  device const uchar* memU8 = memU8_;",
                "  device const float* lut = lut_;",
                "  device const float* inF32 = inF32_;",
                "  threadgroup float* tgF32 = tgF32_;",
                "  const uint gLid = lid, gTgid = tgid, gNtg = ntg, gTgs = tgs;",
                "  uint gbEpoch = 0u;",
                "  const bool vact = true;",
                "  float R[32];",
                "  for (int i=0;i<32;++i) R[i]=0.f;",
                "  R[0] = (float)tid;",
                "  uint U[32];",
                "  for (int i=0;i<32;++i) U[i]=0u;"]
    return """
#include <metal_stdlib>
using namespace metal;

// Auto-JIT (one-dispatch persistent kernel) from bytecode — do not edit
kernel void """ & kernelName & """(
    device float*       memF32_ [[buffer(1)]],
    device const uchar* memU8_  [[buffer(2)]],
    device const float* lut_    [[buffer(3)]],
    device const float* inF32_  [[buffer(4)]],
    threadgroup float*  tgF32_  [[threadgroup(0)]],
    uint tid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg  [[threadgroups_per_grid]],
    uint tgs  [[threads_per_threadgroup]])
{
""" & hdr.join("\n") & "\n" & st.body.join("\n") & """
}
"""
  # U[]/inF32 are declared only when the program REALLY uses them (flag set while decoding,
  # so no separate pre-scan) -> the existing kernels do not change by a single character.
  let uDecl = if st.usesU:
      @["  uint U[32];", "  for (int i=0;i<32;++i) U[i]=0u;"]
    else: newSeq[string]()
  let lines = @["  float R[32];",
                "  for (int i=0;i<32;++i) R[i]=0.f;",
                "  R[0] = (float)tid;"] & uDecl & st.body
  result = """
#include <metal_stdlib>
using namespace metal;

// Auto-JIT from bytecode — do not edit
kernel void """ & kernelName & """(
    device float*       memF32 [[buffer(1)]],
    device const uchar* memU8  [[buffer(2)]],
    device const float* lut    [[buffer(3)]],
""" & (if st.usesIn: "    device const float* inF32  [[buffer(4)]],\n" else: "") & """    threadgroup float*  tgF32  [[threadgroup(0)]],
    uint tid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
""" & lines.join("\n") & """
}
"""

proc lowerBytecodeToGlsl*(p: VmProgram; kernelName: string): string =
  ## Static Vulkan counterpart to the Metal bytecode JIT. In contrast to the
  ## generic interpreter, this emits barriers at uniform static control-flow
  ## locations, which is required by Vulkan when per-lane IF/loop paths diverge.
  var st = LowerState(indent: 2, glsl: true)
  st.lowerAll(p)
  st.closeBlocks()
  let uDecl = if st.usesU:
      @[
        "  uint U[32];",
        "  for (int i=0; i<32; ++i) U[i]=0u;"
      ]
    else: newSeq[string]()
  let lines = @[
    "  float R[32];",
    "  for (int i=0; i<32; ++i) R[i]=0.0;",
    "  R[0] = float(tid);"
  ] & uDecl & st.body
  result = "// Vulkan micro-VM JIT: " & kernelName & "\n" & """
#version 450
layout(local_size_x_id = 0) in;
layout(set=0, binding=0, std430) buffer MemF32Buf { float memF32[]; };
layout(set=0, binding=1, std430) readonly buffer MemU8Buf { uint memU8[]; };
layout(set=0, binding=2, std430) readonly buffer LutBuf { float lut[]; };
layout(set=0, binding=3, std430) readonly buffer InputBuf { float inF32[]; };
layout(set=0, binding=4, std430) buffer MemU32Buf { uint memU32[]; };
layout(constant_id = 1) const uint TG_FLOATS = 1024;
shared float tgF32[TG_FLOATS];
void atomicAddFloat(uint idx, float value) {
  uint oldBits = memU32[idx];
  for (int spin=0; spin<4096; ++spin) {
    uint newBits = floatBitsToUint(uintBitsToFloat(oldBits) + value);
    uint observed = atomicCompSwap(memU32[idx], oldBits, newBits);
    if (observed == oldBits) return;
    oldBits = observed;
  }
}
void main() {
  uint tid = gl_GlobalInvocationID.x;
  uint lid = gl_LocalInvocationID.x;
  uint tgid = gl_WorkGroupID.x;
""" & lines.join("\n") & "\n}\n"
  result = result.replace("as_type<float>(", "uintBitsToFloat(")
  result = result.replace("rsqrt(", "inversesqrt(")

var gGlslByKernel = initTable[string, string]()

proc lowerBytecodeToGlslCached*(p: VmProgram; kernelName: string): string =
  ## Lowering is a pure function of (bytecode, kernel name) and costs
  ## milliseconds for the scan/conv programs (thousands of emitted lines).
  ## Inference encodes the same program on every layer, so cache by kernel
  ## name: that name already embeds a hash of the program words.
  if kernelName in gGlslByKernel: return gGlslByKernel[kernelName]
  result = lowerBytecodeToGlsl(p, kernelName)
  gGlslByKernel[kernelName] = result

# ── Program key: hash computed by a MICRO-KERNEL on the GPU ────────
## Before: `for w in p.words: h = h !& hash(w)` on the CPU. Now: a bytecode
## program (buildHashProgram) with FOR_BEGIN/END runs on the GPU, reads the program
## bytes and accumulates 2 independent uint32 hashes in U[] registers -> a 64-bit key.
##
## Buffer layout (the same float32 buffer bound to slots 1, 2 and 3):
##   memF32[0]    = program size in bytes
##   memF32[1..4] = OUT: lo16(h1), hi16(h2)... (each uint32 split into two 16-bit halves because
##                  R[] is float32 and holds integers exactly only up to 2^24)
##   from byte offset 32 (= float 8..) = the program words (memU8 view)

const
  HashHeaderFloats = 8                ## 32 byte header
  HashKernelName = "jit_vm_hash"

proc buildHashProgram*(): VmProgram =
  var p: VmProgram
  p.emitLoadF32(1, 0, 0)                  # R1 = nBytes   (tid 0 only)
  p.emitUConst(0, 0x811C9DC5'u32)         # U0 = h1
  p.emitUConst(1, 0x9E3779B1'u32)         # U1 = h2
  p.emitUConst(2, uint32(HashHeaderFloats * 4))  # U2 = byte pointer
  p.emitForBegin(1)                       # for _ in 0 ..< nBytes   (GPU)
  p.emitULoadU8(3, 2, 0)                  #   U3 = memU8[U2]
  p.emitUAdd(0, 0, 3)
  p.emitUMulK(0, 0, 0x01000193'u32)       #   h1 = (h1 + byte) * FNV prime
  p.emitUAdd(1, 1, 3)
  p.emitUMulK(1, 1, 0x85EBCA6B'u32)       #   h2 = (h2 + byte) * murmur c
  p.emitUAddK(2, 2, 1)                    #   U2 += 1
  p.emitForEnd()
  # write the 2 hashes as 4 16-bit halves (exact in float)
  p.emitUAndK(4, 0, 0xFFFF'u32); p.emitUToR(2, 4); p.emitStoreF32(2, 0, 1)
  p.emitUShrK(4, 0, 16'u32);     p.emitUToR(2, 4); p.emitStoreF32(2, 0, 2)
  p.emitUAndK(4, 1, 0xFFFF'u32); p.emitUToR(2, 4); p.emitStoreF32(2, 0, 3)
  p.emitUShrK(4, 1, 16'u32);     p.emitUToR(2, 4); p.emitStoreF32(2, 0, 4)
  p.emitHalt()
  result = p

var gHashKernelReady = false

var gWordsKey = initTable[seq[uint32], string]()
  ## key computed from the bytecode CONTENT (exact word match) -> each program is hashed only ONCE.

proc fallbackWordsKey(words: seq[uint32]): string =
  ## Key that needs NO GPU/session: used while INSIDE a session (Metal does not allow nested sessions,
  ## so a separate session for GPU hashing is impossible). std hash + length; prefix 'S' to tell it apart.
  "S" & toHex(uint64(words.len), 4) & toHex(cast[uint64](hash(words)), 16)

proc sanitizeTag(tag: string): string =
  ## keeps only [A-Za-z0-9_], other characters -> '_' (peg, no explicit loop)
  tag.replace(peg"[^A-Za-z0-9_]", "_")

when defined(macosx):
  var jitHashBackend* = byby.gbAuto   ## change if another backend than gbAuto is needed
  # tryMetalJitCompile is declared below; forward-declared for the hash kernel
  proc tryMetalJitCompile*(source, kernelName: string): bool

  proc gpuHashWords(words: seq[uint32]): string =
    if not gHashKernelReady:
      let src = lowerBytecodeToMetal(buildHashProgram(), HashKernelName)
      if not tryMetalJitCompile(src, HashKernelName):
        raise newException(CatchableError, "jit_vm: compile hash micro-kernel THAT BAI")
      gHashKernelReady = true
    let nBytes = words.len * 4
    # memcpy (not an arithmetic loop): words -> the float32 region after the header
    var host = newSeq[float32](HashHeaderFloats + words.len)
    host[0] = float32(nBytes)
    if words.len > 0:
      copyMem(addr host[HashHeaderFloats], unsafeAddr words[0], nBytes)
    let be = jitHashBackend
    if not byby.sessionBegin(be):
      # INSIDE a session (Metal refuses nested sessions): cannot hash on the GPU now.
      # Use the fallback key (no GPU needed) instead of raising - the result is memoized by program content.
      return fallbackWordsKey(words)
    let h = byby.sessionUploadPrivate(be, host)
    var ok = sessionPipeline(HashKernelName)
    ok = ok and sessionBindBuffer(1, h.metalHandle) and
         sessionBindBuffer(2, h.metalHandle) and sessionBindBuffer(3, h.metalHandle)
    ok = ok and sessionSetThreadgroupBytes(0, 16) and sessionDispatch(1, 1)
    let ended = byby.sessionEnd(be)
    if not (ok and ended):
      raise newException(CatchableError, "jit_vm: dispatch hash micro-kernel that bai")
    let o = byby.sessionRead(be, h, HashHeaderFloats)
    if o.len < 5:
      raise newException(CatchableError, "jit_vm: reading the hash result from the GPU failed")
    let h1 = uint64(uint32(o[1])) or (uint64(uint32(o[2])) shl 16)
    let h2 = uint64(uint32(o[3])) or (uint64(uint32(o[4])) shl 16)
    result = toHex((h2 shl 32) or h1, 16)
else:
  proc gpuHashWords(words: seq[uint32]): string =
    ## Builds outside macOS (only for nim check/bytecode tests): NO GPU, so no
    ## hashing. O(1) key (length + first/last word), no loop. Safe because
    ## this branch never compiles/dispatches a kernel (tryMetalJitCompile = false).
    let first = if words.len > 0: words[0] else: 0'u32
    let last = if words.len > 0: words[words.len - 1] else: 0'u32
    toHex(uint64(words.len), 8) & toHex(uint64(first), 8)[^4 .. ^1] & toHex(uint64(last), 8)[^4 .. ^1]

var gJitDeviceReady = false

proc ensureJitDeviceReady*(): bool =
  ## Initialises the GPU device (Metal: gDevice/gQueue) before any JIT/hash kernel. Safe to call repeatedly.
  ## Uses the existing API: sessionBegin -> metalSessionBegin -> ensureMetalInit (dispatch_once).
  ## If sessionBegin returns false because a session IS open, the device is surely initialised -> ready.
  when defined(macosx):
    if not gJitDeviceReady:
      gJitDeviceReady = true     # tried only once (false = inside a session => device already initialised)
      if byby.sessionBegin(byby.gbAuto):
        discard byby.sessionEnd(byby.gbAuto)
    result = true
  else:
    result = true

proc programKey*(p: VmProgram; tag = ""): string =
  ## Hash (GPU micro-kernel) + tag. The name must be a valid Metal identifier.
  result = gWordsKey.getOrDefault(p.words, "")
  if result.len == 0:
    result = gpuHashWords(p.words)       # GPU (opens its own session) - inside a session: fallback key
    gWordsKey[p.words] = result
  if tag.len > 0:
    result.add '_'
    result.add sanitizeTag(tag)

proc programKeyNoSession*(p: VmProgram; tag = ""): string =
  ## LIKE programKey but NEVER opens a session (safe INSIDE a session, e.g. encodeMicroVm).
  ## A program hashed on the GPU before reuses that key (same kernel name); otherwise the fallback.
  result = gWordsKey.getOrDefault(p.words, "")
  if result.len == 0:
    result = fallbackWordsKey(p.words)
    gWordsKey[p.words] = result
  if tag.len > 0:
    result.add '_'
    result.add sanitizeTag(tag)

# ── Compile hook (Metal) — needs a C bridge ──────────────────────
## Like tinygrad's MetalCompiler: source → compiled library → pipeline name.
## The shim needs (once):
##   int metal_jit_compile(const char* src, const char* kernelName);
##   // uses newLibraryWithSource:options:error: then merges the function into the pipeline cache
##
## Without the shim: lowering still runs (source visible), compile returns false → stays on the interpreter.

when defined(macosx):
  proc metal_jit_compile(source, kernelName: cstring): cint {.
    importc: "metal_jit_compile", header: "metal_shim.h".}

  proc metalJitCompileAvailable*: bool = true

  proc tryMetalJitCompile*(source, kernelName: string): bool =
    try:
      # FIX "[metal_jit] gDevice nil": make sure Metal is initialised (gDevice/gQueue) BEFORE JIT.
      if not ensureJitDeviceReady():
        stderr.writeLine "[jit_vm] GPU (Metal) initialisation failed - cannot JIT compile"
        return false
      result = metal_jit_compile(source.cstring, kernelName.cstring) != 0
    except:
      result = false
else:
  ## Builds outside macOS (e.g. Linux CI only for nim check/bytecode tests): no
  ## metal_shim.h/.m on the include path. Compiles "successfully" but does
  ## nothing - every kernel stays on the interpret tier (session_vm_dispatch also stubs
  ## sessionPipeline/... = true in its else branch).
  proc metalJitCompileAvailable*: bool = false
  proc tryMetalJitCompile*(source, kernelName: string): bool = false

# ── Main API: run a program through the cache + JIT ──────────────────

proc noteHit*(cache: var JitCache; key: string): var CompiledEntry =
  if key notin cache.entries:
    cache.entries[key] = CompiledEntry(tier: tierInterpret, hitCount: 0,
                                        kernelName: "micro_vm_kernel")
  cache.entries[key].hitCount.inc
  result = cache.entries[key]

proc maybeJit*(cache: var JitCache; p: VmProgram; key: string): CompiledEntry =
  ## Call before each run. When hot enough and Metal JIT is on → compile + move to tierCompiled.
  var e = cache.noteHit(key)
  if e.tier == tierCompiled:
    return e
  if not cache.cfg.enableMetalJit:
    e.tier = tierProfile
    cache.entries[key] = e
    return e
  if e.hitCount < cache.cfg.hotThreshold:
    e.tier = tierProfile
    cache.entries[key] = e
    return e

  # Hot enough → lower + compile
  let kname = "jit_" & key
  let src = lowerBytecodeToMetal(p, kname)
  e.metalSource = src
  e.kernelName = kname
  # NEW: dumps the WHOLE generated Metal source - before, only the
  # kernel NAME was logged ("OK kernel=jit_..."), never the generated
  # CODE. Enabled by an env var (off by default, the source can be long/
  # spammy) - very useful to check that buildSplitProjProgram/buildConv1d.../
  # the VmProgram builders produce sensible Metal code
  # (e.g. a missing threadgroup cache like the LUT bug found earlier in fp4 gemv,
  # correct unrolling/vectorisation, etc).
  if getEnv("BYBY_JIT_DUMP_SOURCE", "0") == "1":
    stderr.writeLine "════ [jit_vm] generated Metal source for kernel=" & kname &
      " (key=" & key & ") ════"
    stderr.writeLine src
    stderr.writeLine "════ [jit_vm] het source " & kname & " ════"
  if tryMetalJitCompile(src, kname):
    e.tier = tierCompiled
    e.compiledAt = epochTime()
  else:
    # No bridge / compile failed → stay on the interpreter, keep the source for debugging
    e.tier = tierProfile
    e.kernelName = "micro_vm_kernel"
  cache.entries[key] = e
  result = e

## Bytecode hashes are memoized by their complete word sequence in gWordsKey.
## A tag is a readable namespace, not proof that programs with that tag match.

proc dispatchProgramTagged*(cache: var JitCache; p: VmProgram; tag: string): string =
  ## Always include the full program hash. The old tag-only cache was unsound:
  ## equal-shape kernels with different embedded buffer offsets could silently
  ## reuse a pipeline compiled for the first layout.
  let key = programKey(p, tag)
  result = cache.maybeJit(p, key).kernelName

proc dispatchProgramNoSession*(cache: var JitCache; p: VmProgram; tag = ""): string =
  ## Like dispatchProgram but never opens a session to hash -> usable INSIDE an open session
  ## (encodeMicroVm/encodeVm while encoding). JIT compile (when hot) needs no session.
  let key = programKeyNoSession(p, tag)
  cache.maybeJit(p, key).kernelName

proc dispatchProgram*(cache: var JitCache; p: VmProgram; tag = ""): string =
  ## Returns the kernel name for sessionPipeline.
  ## tierCompiled → the JIT kernel; otherwise micro_vm_kernel.
  let key = programKey(p, tag)
  let e = cache.maybeJit(p, key)
  result = e.kernelName

# ── Vulkan GLSL helpers (multi-buffer inference path) ─────────

proc kernelNameFor*(cache: var JitCache; p: VmProgram; tag: string): string =
  ## Compile GLSL, cache SPIR-V name, return kernelName. No dispatch.
  ## Used at prepare time so hot-path encode only looks up cached source.
  let key = programKeyNoSession(p, tag)
  var e = cache.noteHit(key)
  if e.glslSource.len == 0 or e.kernelName == "micro_vm_kernel" or e.kernelName.len == 0:
    let kname = "jit_vk_" & key
    let src = lowerBytecodeToGlsl(p, kname)
    e.glslSource = src
    e.kernelName = kname
    e.tier = tierCompiled
    e.compiledAt = epochTime()
    # Logs a NEW compile only with BYBY_VULKAN_JIT_VERBOSE: it used to print
    # unconditionally on every compile -> "Vulkan JIT" spam made the log unreadable.
    # The cache-hit branch has its own elif below (same BYBY_VULKAN_JIT_VERBOSE gate).
    if getEnv("BYBY_VULKAN_JIT_VERBOSE", "0") == "1":
      stderr.writeLine "[vulkan_jit] kernelNameFor tag=" & tag &
        " kernel=" & kname & " srcBytes=" & $src.len & " words=" & $p.words.len
    if getEnv("BYBY_JIT_DUMP_SOURCE", "0") == "1":
      stderr.writeLine "════ [vulkan_jit] GLSL source kernel=" & kname & " ════"
      stderr.writeLine src
      stderr.writeLine "════ [vulkan_jit] het source " & kname & " ════"
    cache.entries[key] = e
  elif getEnv("BYBY_VULKAN_JIT_VERBOSE", "0") == "1":
    stderr.writeLine "[vulkan_jit] kernelNameFor CACHE tag=" & tag &
      " kernel=" & e.kernelName
  result = e.kernelName

proc glslSourceFor*(cache: var JitCache; p: VmProgram; tag: string): string =
  ## Return GLSL source string for program (generate + cache if needed).
  let key = programKeyNoSession(p, tag)
  discard cache.kernelNameFor(p, tag)
  result = cache.entries[key].glslSource

proc cachedGlslSource*(cache: var JitCache; kernelName: string): string =
  ## Lookup GLSL source by kernel name (after kernelNameFor has run once).
  for e in cache.entries.values:
    if e.kernelName == kernelName:
      return e.glslSource
  result = ""

# ── Merging programs (merge files) ─────────────────────────────────
## Plain data concatenation (no arithmetic), written recursively - no for/while.
## BUG FIX vs the old version: it dropped EVERY word whose low byte == HALT (0x01), including
## OPERAND words (e.g. OP_U_CONST ... 1, OP_U_ADDK ... 1, float constants with low byte 1)
## -> silently corrupting the program. Now decoded properly: only real HALT opcodes are dropped.

proc mergePart(dst: var VmProgram; part: VmProgram; pc: int; dropHalt: bool) =
  if pc >= part.words.len: return
  let w = part.words[pc]
  let op = w and 0xFF'u32
  if op == OP_HALT and dropHalt:
    dst.mergePart(part, pc + 1, dropHalt)
  elif opHasWord(op) and pc + 1 < part.words.len:
    dst.words.add w
    dst.words.add part.words[pc + 1]
    dst.mergePart(part, pc + 2, dropHalt)
  else:
    dst.words.add w
    dst.mergePart(part, pc + 1, dropHalt)

proc mergeParts(dst: var VmProgram; parts: openArray[VmProgram]; i: int) =
  if i >= parts.len: return
  dst.mergePart(parts[i], 0, i < parts.high)
  dst.mergeParts(parts, i + 1)

proc mergePrograms*(parts: openArray[VmProgram]): VmProgram =
  ## Merges several .bvm files into one that runs in a single dispatch (mid-program HALTs dropped).
  result.mergeParts(parts, 0)
  if result.words.len == 0 or (result.words[^1] and 0xFF'u32) != OP_HALT:
    result.emitHalt()

# ── Demo ─────────────────────────────────────────────────────

proc demoRun(cache: var JitCache; prog: VmProgram; i, last: int) =
  if i > last: return
  echo "run#", i, " kernel=", cache.dispatchProgram(prog, "demo")
  cache.demoRun(prog, i + 1, last)

when isMainModule:
  var cache = initJitCache()
  var prog = buildExampleLutFma()
  echo "program words=", prog.words.len
  cache.demoRun(prog, 1, 12)
  let key = programKey(prog, "demo")
  if key in cache.entries:
    let e = cache.entries[key]
    echo "tier=", e.tier, " hits=", e.hitCount
    if e.metalSource.len > 0:
      let ls = e.metalSource.splitLines()
      echo "---- JIT Metal source (head) ----"
      echo ls[0 .. min(20, ls.high)].join("\n")
