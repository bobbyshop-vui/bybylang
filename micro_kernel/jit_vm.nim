## jit_vm.nim — JIT bytecode theo HotSpot + tinygrad MetalCompiler
##
## Pipeline (đúng tài liệu bạn gửi):
##
##   Bytecode (.bvm / VmProgram)
##        │
##        ├─► Interpreter (micro_vm_kernel) + đếm hot
##        │
##        └─► khi hot ≥ threshold
##                 ├─► lower bytecode → Metal source (string)
##                 ├─► Metal compile (newLibraryWithSource / MTLCompiler)
##                 ├─► code cache (key = hash program + meta)
##                 └─► lần sau: dispatch kernel đã JIT, bỏ interpreter
##
## Kernel cũ (matmul_q4, vecop, ...) GIỮ NGUYÊN — path nóng production.
## JIT chỉ cho program micro-VM / logic custom.

import std/[tables, strutils, strformat, times, os, pegs, hashes]
import ./session_vm
when defined(macosx):
  import ./session_vm_dispatch
  # backend GPU sống ở vendor/bybylang/gpubackend (cùng package bybylang)
  import ../gpubackend as byby

# ── Cấu hình tier (HotSpot-style) ────────────────────────────

type
  JitTier* = enum
    tierInterpret = 0  ## chỉ micro_vm
    tierProfile   = 1  ## interpret + đếm
    tierCompiled  = 2  ## đã có Metal pipeline từ JIT

  JitConfig* = object
    hotThreshold*: int     ## số lần chạy trước khi JIT (mặc định 10)
    enableMetalJit*: bool  ## compile ra Metal source
    enableVulkanJit*: bool ## compile ra GLSL source (Vulkan)
    enableCpuJit*: bool    ## (dự phòng) LLVM ORC — chưa nối
    maxCacheEntries*: int

  CompiledEntry* = object
    tier*: JitTier
    hitCount*: int
    metalSource*: string   ## source đã lower (debug / tái compile)
    glslSource*: string    ## GLSL source đã lower (Vulkan path)
    kernelName*: string    ## tên trong library sau JIT
    compiledAt*: float
    ## pipeline handle thật nằm phía C/Metal; Nim giữ tên + key

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
## KHÔNG còn vòng for/while nào trên host: bộ giải mã chạy bằng đệ quy đuôi.
## (Sinh TEXT Metal bắt buộc nằm ở host - GPU không tự viết source cho chính nó -
## nhưng đó là xử lý chuỗi, không phải phép tính số. Mọi phép tính số (hash...)
## đã chuyển sang micro-kernel, xem programKey.)
## Kernel cũ sinh ra Y HỆT như trước (đã diff với bản gốc).

type LowerState = object
  body: seq[string]
  pc, loopDepth, indent: int
  usesU, usesIn: bool
  glsl: bool
  od: bool              ## dung op one-dispatch (persistent kernel) -> kernel co prologue rieng
  vtgModes: seq[uint32] ## stack mode cua cac khoi OP_VTG_BEGIN dang mo

proc ind(st: LowerState): string = "  ".repeat(st.indent)

proc imm(st: var LowerState; p: VmProgram): uint32 =
  ## Đọc word toán hạng kế tiếp (0 nếu hết program) và nhảy qua nó.
  if st.pc < p.words.len:
    result = p.words[st.pc]
    inc st.pc

proc lowerStep(st: var LowerState; p: VmProgram) =
  let w = p.words[st.pc]
  inc st.pc
  let op = w and 0xFF'u32
  let a = (w shr 8) and 0xF
  let b = (w shr 12) and 0xF
  let c = (w shr 16) and 0xF
  if op >= OP_U_TGID and op <= OP_U_ATOMIC_ADD: st.usesU = true
  if op == OP_U_LOAD_IN: st.usesIn = true
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
    let rc = st.imm(p) and 0xF
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
    inc st.loopDepth   # dùng chung bộ đếm đóng-ngoặc với FOR (đủ vì chỉ cần biết còn khối nào chưa đóng ở cuối hàm)
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
      st.body.add st.ind() & "/* OP_GRID_BARRIER chua ho tro GLSL */ return;"
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
      # V=32 < tgs: moi TG vat ly chay tgs/32 threadgroup ao song song (sub), moi sub 1 vung tgF32 rieng.
      # So vong LAP CO DINH (dong deu moi TG) de cac barrier ben trong khong lech.
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
      st.body.add st.ind() & "    float R[16]; for (int i=0;i<16;++i) R[i]=0.f; R[0] = (float)tid;"
      st.body.add st.ind() & "    uint U[16]; for (int i=0;i<16;++i) U[i]=0u;"
      st.indent += 2
    else:
      st.body.add st.ind() & &"for (uint vIt = gTgid; vIt < {nVirt}u; vIt += gNtg) {{"
      st.body.add st.ind() & "  const uint tgid = vIt;"
      st.body.add st.ind() & "  const uint lid = gLid;"
      st.body.add st.ind() & "  const uint tid = vIt * gTgs + gLid;"
      st.body.add st.ind() & "  const bool vact = true;"
      st.body.add st.ind() & "  threadgroup float* tgF32 = tgF32_;"
      st.body.add st.ind() & "  float R[16]; for (int i=0;i<16;++i) R[i]=0.f; R[0] = (float)tid;"
      st.body.add st.ind() & "  uint U[16]; for (int i=0;i<16;++i) U[i]=0u;"
      st.indent += 1
      if a == 2:
        # mode guard: chi chay than lenh neu tid < NT (than KHONG duoc chua barrier_tg)
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
  ## đệ quy đuôi thay cho `while pc < n`
  if st.pc >= p.words.len: return
  st.lowerStep(p)
  st.lowerAll(p)

proc closeBlocks(st: var LowerState) =
  ## đóng nốt các khối FOR/IF còn mở (thay cho `while loopDepth > 0`)
  if st.loopDepth <= 0: return
  dec st.indent
  dec st.loopDepth
  st.body.add st.ind() & "}"
  st.closeBlocks()

proc lowerBytecodeToMetal*(p: VmProgram; kernelName: string): string =
  ## Sinh 1 kernel Metal từ bytecode. Mọi thread: tid → R0.
  var st = LowerState(indent: 2)
  st.lowerAll(p)
  st.closeBlocks()
  if st.od:
    ## ── ONE-DISPATCH persistent kernel ───────────────────────────────────
    ## Ten goc *_ giu nguyen buffer; memF32/memU8/lut/inF32 la con tro CO THE doi base
    ## (OP_BASE_*). tid/lid/tgid ben trong khoi VTG bi che (shadow) boi gia tri AO.
    let hdr = @["  device float* memF32 = memF32_;",
                "  device const uchar* memU8 = memU8_;",
                "  device const float* lut = lut_;",
                "  device const float* inF32 = inF32_;",
                "  threadgroup float* tgF32 = tgF32_;",
                "  const uint gLid = lid, gTgid = tgid, gNtg = ntg, gTgs = tgs;",
                "  uint gbEpoch = 0u;",
                "  const bool vact = true;",
                "  float R[16];",
                "  for (int i=0;i<16;++i) R[i]=0.f;",
                "  R[0] = (float)tid;",
                "  uint U[16];",
                "  for (int i=0;i<16;++i) U[i]=0u;"]
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
  # Khai báo U[]/inF32 chỉ khi program THẬT SỰ dùng (cờ đặt ngay lúc giải mã,
  # nên không cần pre-scan riêng) -> kernel cũ không đổi một ký tự.
  let uDecl = if st.usesU:
      @["  uint U[16];", "  for (int i=0;i<16;++i) U[i]=0u;"]
    else: newSeq[string]()
  let lines = @["  float R[16];",
                "  for (int i=0;i<16;++i) R[i]=0.f;",
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
        "  uint U[16];",
        "  for (int i=0; i<16; ++i) U[i]=0u;"
      ]
    else: newSeq[string]()
  let lines = @[
    "  float R[16];",
    "  for (int i=0; i<16; ++i) R[i]=0.0;",
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

# ── Key program: hash chạy bằng MICRO-KERNEL trên GPU ────────
## Trước đây: `for w in p.words: h = h !& hash(w)` trên CPU. Giờ: 1 program
## bytecode (buildHashProgram) có FOR_BEGIN/END chạy trên GPU, đọc byte của
## program, tích luỹ 2 hash uint32 độc lập bằng thanh ghi U[] -> key 64-bit.
##
## Layout buffer (cùng 1 buffer float32 bind vào slot 1, 2 và 3):
##   memF32[0]    = số byte của program
##   memF32[1..4] = OUT: lo16(h1), hi16(h2)... (mỗi uint32 tách 2 nửa 16-bit vì
##                  R[] là float32 chỉ giữ chính xác số nguyên <= 2^24)
##   byte offset 32 trở đi (= float 8..) = các word của program (memU8 view)

const
  HashHeaderFloats = 8                ## 32 byte header
  HashKernelName = "jit_vm_hash"

proc buildHashProgram*(): VmProgram =
  var p: VmProgram
  p.emitLoadF32(1, 0, 0)                  # R1 = nBytes   (chỉ tid 0 chạy)
  p.emitUConst(0, 0x811C9DC5'u32)         # U0 = h1
  p.emitUConst(1, 0x9E3779B1'u32)         # U1 = h2
  p.emitUConst(2, uint32(HashHeaderFloats * 4))  # U2 = con trỏ byte
  p.emitForBegin(1)                       # for _ in 0 ..< nBytes   (GPU)
  p.emitULoadU8(3, 2, 0)                  #   U3 = memU8[U2]
  p.emitUAdd(0, 0, 3)
  p.emitUMulK(0, 0, 0x01000193'u32)       #   h1 = (h1 + byte) * FNV prime
  p.emitUAdd(1, 1, 3)
  p.emitUMulK(1, 1, 0x85EBCA6B'u32)       #   h2 = (h2 + byte) * murmur c
  p.emitUAddK(2, 2, 1)                    #   U2 += 1
  p.emitForEnd()
  # ghi 2 hash dưới dạng 4 nửa 16-bit (float chính xác)
  p.emitUAndK(4, 0, 0xFFFF'u32); p.emitUToR(2, 4); p.emitStoreF32(2, 0, 1)
  p.emitUShrK(4, 0, 16'u32);     p.emitUToR(2, 4); p.emitStoreF32(2, 0, 2)
  p.emitUAndK(4, 1, 0xFFFF'u32); p.emitUToR(2, 4); p.emitStoreF32(2, 0, 3)
  p.emitUShrK(4, 1, 16'u32);     p.emitUToR(2, 4); p.emitStoreF32(2, 0, 4)
  p.emitHalt()
  result = p

var gHashKernelReady = false

var gWordsKey = initTable[seq[uint32], string]()
  ## key da tinh theo NOI DUNG bytecode (khop chinh xac tung word) -> moi chuong trinh chi hash MOT lan.

proc fallbackWordsKey(words: seq[uint32]): string =
  ## Key KHONG can GPU/session: dung khi DANG trong session (Metal khong cho session long nhau nen
  ## khong the mo session rieng de hash tren GPU). hash cua std + do dai; tien to 'S' de phan biet.
  "S" & toHex(uint64(words.len), 4) & toHex(cast[uint64](hash(words)), 16)

proc sanitizeTag(tag: string): string =
  ## chỉ giữ [A-Za-z0-9_], ký tự khác -> '_' (peg, không vòng lặp tường minh)
  tag.replace(peg"[^A-Za-z0-9_]", "_")

when defined(macosx):
  var jitHashBackend* = byby.gbAuto   ## đổi nếu cần backend khác gbAuto
  # tryMetalJitCompile khai báo ở dưới; forward declare để dùng cho hash kernel
  proc tryMetalJitCompile*(source, kernelName: string): bool

  proc gpuHashWords(words: seq[uint32]): string =
    if not gHashKernelReady:
      let src = lowerBytecodeToMetal(buildHashProgram(), HashKernelName)
      if not tryMetalJitCompile(src, HashKernelName):
        raise newException(CatchableError, "jit_vm: compile hash micro-kernel THAT BAI")
      gHashKernelReady = true
    let nBytes = words.len * 4
    # memcpy (không phải vòng tính toán): word -> vùng float32 sau header
    var host = newSeq[float32](HashHeaderFloats + words.len)
    host[0] = float32(nBytes)
    if words.len > 0:
      copyMem(addr host[HashHeaderFloats], unsafeAddr words[0], nBytes)
    let be = jitHashBackend
    if not byby.sessionBegin(be):
      # DANG trong session (Metal tu choi session long nhau): khong the hash tren GPU luc nay.
      # Dung key fallback (khong can GPU) thay vi raise - ket qua duoc nho lai theo noi dung program.
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
      raise newException(CatchableError, "jit_vm: doc ket qua hash tu GPU that bai")
    let h1 = uint64(uint32(o[1])) or (uint64(uint32(o[2])) shl 16)
    let h2 = uint64(uint32(o[3])) or (uint64(uint32(o[4])) shl 16)
    result = toHex((h2 shl 32) or h1, 16)
else:
  proc gpuHashWords(words: seq[uint32]): string =
    ## Build ngoài macOS (chỉ để nim check/test bytecode): KHÔNG có GPU nên không
    ## hash được. Key O(1) (độ dài + word đầu/cuối), không vòng lặp. An toàn vì
    ## nhánh này không bao giờ compile/dispatch kernel (tryMetalJitCompile = false).
    let first = if words.len > 0: words[0] else: 0'u32
    let last = if words.len > 0: words[words.len - 1] else: 0'u32
    toHex(uint64(words.len), 8) & toHex(uint64(first), 8)[^4 .. ^1] & toHex(uint64(last), 8)[^4 .. ^1]

var gJitDeviceReady = false

proc ensureJitDeviceReady*(): bool =
  ## Khoi tao device GPU (Metal: gDevice/gQueue) truoc moi JIT/hash kernel. An toan goi nhieu lan.
  ## Dung API co san: sessionBegin -> metalSessionBegin -> ensureMetalInit (dispatch_once).
  ## Neu sessionBegin tra false vi DANG co session mo thi device chac chan da init -> coi la san sang.
  when defined(macosx):
    if not gJitDeviceReady:
      gJitDeviceReady = true     # chi thu 1 lan (false = dang trong session => device da init)
      if byby.sessionBegin(byby.gbAuto):
        discard byby.sessionEnd(byby.gbAuto)
    result = true
  else:
    result = true

proc programKey*(p: VmProgram; tag = ""): string =
  ## Hash (GPU micro-kernel) + tag. Tên phải là identifier Metal hợp lệ.
  result = gWordsKey.getOrDefault(p.words, "")
  if result.len == 0:
    result = gpuHashWords(p.words)       # GPU (mo session rieng) - neu dang trong session: key fallback
    gWordsKey[p.words] = result
  if tag.len > 0:
    result.add '_'
    result.add sanitizeTag(tag)

proc programKeyNoSession*(p: VmProgram; tag = ""): string =
  ## NHU programKey nhung KHONG BAO GIO mo session (an toan khi DANG trong session, vd encodeMicroVm).
  ## Chuong trinh da hash tren GPU truoc do thi dung lai key do (cung ten kernel); chua co thi dung fallback.
  result = gWordsKey.getOrDefault(p.words, "")
  if result.len == 0:
    result = fallbackWordsKey(p.words)
    gWordsKey[p.words] = result
  if tag.len > 0:
    result.add '_'
    result.add sanitizeTag(tag)

# ── Compile hook (Metal) — cần bridge C ──────────────────────
## Giống tinygrad MetalCompiler: source → compiled library → pipeline name.
## Shim cần thêm (một lần):
##   int metal_jit_compile(const char* src, const char* kernelName);
##   // dùng newLibraryWithSource:options:error: rồi merge function vào cache pipeline
##
## Khi chưa có shim: lower vẫn chạy (xem source), compile trả false → ở lại interpreter.

when defined(macosx):
  proc metal_jit_compile(source, kernelName: cstring): cint {.
    importc: "metal_jit_compile", header: "metal_shim.h".}

  proc metalJitCompileAvailable*: bool = true

  proc tryMetalJitCompile*(source, kernelName: string): bool =
    try:
      # FIX "[metal_jit] gDevice nil": dam bao Metal da init (gDevice/gQueue) TRUOC khi JIT.
      if not ensureJitDeviceReady():
        stderr.writeLine "[jit_vm] khoi tao GPU(Metal) that bai - khong the JIT compile"
        return false
      result = metal_jit_compile(source.cstring, kernelName.cstring) != 0
    except:
      result = false
else:
  ## Build ngoai macOS (vd CI Linux chi de nim check/test bytecode): khong co
  ## metal_shim.h/.m tren include path. Compile "thanh cong" nhung khong lam
  ## gi - moi kernel roi ve tier interpret (session_vm_dispatch cung da stub
  ## sessionPipeline/... = true tren nhanh else cua no).
  proc metalJitCompileAvailable*: bool = false
  proc tryMetalJitCompile*(source, kernelName: string): bool = false

# ── API chính: chạy program qua cache + JIT ──────────────────

proc noteHit*(cache: var JitCache; key: string): var CompiledEntry =
  if key notin cache.entries:
    cache.entries[key] = CompiledEntry(tier: tierInterpret, hitCount: 0,
                                        kernelName: "micro_vm_kernel")
  cache.entries[key].hitCount.inc
  result = cache.entries[key]

proc maybeJit*(cache: var JitCache; p: VmProgram; key: string): CompiledEntry =
  ## Gọi trước mỗi lần chạy. Nếu đủ nóng và bật Metal JIT → compile + lên tierCompiled.
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

  # Đủ nóng → lower + compile
  let kname = "jit_" & key
  let src = lowerBytecodeToMetal(p, kname)
  e.metalSource = src
  e.kernelName = kname
  # MOI: dump TOAN BO Metal source THAT SU duoc sinh ra - truoc gio chi
  # log TEN kernel ("OK kernel=jit_..."), khong bao gio thay NOI DUNG code
  # duoc gen ra sao. Bat bang env var (mac dinh TAT vi source co the dai/
  # spam) - cuc ky huu ich de kiem tra buildSplitProjProgram/buildConv1d.../
  # cac ham sinh VmProgram co dang ra dung Metal code hop ly hay khong
  # (vi du: co bi thieu threadgroup cache nhu bug LUT o fp4 gemv da tim
  # thay truoc do khong, co unroll/vector hoa dung khong, v.v).
  if getEnv("BYBY_JIT_DUMP_SOURCE", "0") == "1":
    stderr.writeLine "════ [jit_vm] Metal source sinh ra cho kernel=" & kname &
      " (key=" & key & ") ════"
    stderr.writeLine src
    stderr.writeLine "════ [jit_vm] het source " & kname & " ════"
  if tryMetalJitCompile(src, kname):
    e.tier = tierCompiled
    e.compiledAt = epochTime()
  else:
    # Chưa có bridge / compile fail → giữ interpreter, vẫn lưu source để debug
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
  ## Nhu dispatchProgram nhung khong bao gio mo session de hash -> dung duoc TRONG session dang mo
  ## (encodeMicroVm/encodeVm luc encode). JIT compile (neu du nong) khong can session.
  let key = programKeyNoSession(p, tag)
  cache.maybeJit(p, key).kernelName

proc dispatchProgram*(cache: var JitCache; p: VmProgram; tag = ""): string =
  ## Trả về tên kernel nên sessionPipeline.
  ## tierCompiled → kernel JIT; không thì micro_vm_kernel.
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
    # Log compile MOI chi khi bat BYBY_VULKAN_JIT_VERBOSE: truoc day in vo dieu
    # kien moi lan compile -> "Vulkan JIT in tua lua" khong the doc log.
    # Nhanh cache-hit da co elif rieng o duoi (cung gate BYBY_VULKAN_JIT_VERBOSE).
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

# ── Gộp program (merge file) ─────────────────────────────────
## Chỉ là nối dữ liệu (không có phép tính số), viết bằng đệ quy - không for/while.
## SỬA BUG so với bản cũ: bản cũ bỏ MỌI word có byte thấp == HALT (0x01), kể cả
## word TOÁN HẠNG (vd OP_U_CONST ... 1, OP_U_ADDK ... 1, hằng float có byte thấp 1)
## -> làm hỏng program im lặng. Giờ giải mã đúng: chỉ bỏ opcode HALT thật.

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
  ## Gộp nhiều .bvm thành một file chạy một dispatch (bỏ HALT giữa chừng).
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
