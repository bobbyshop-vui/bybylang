## session_vm.nim — approach A: write bytecode (file/program) and run it with micro_vm_kernel
##
## No high-level sessionMatmul composition. "Writing a new kernel" = emitting very low-level ops.
##
## sessionFor = a HOST-side for loop while writing the program (or OP_FOR_BEGIN/END in the bytecode).

import std/streams

# ── Opcodes (must match micro_vm.metal) ─────────────────────────────
const
  OP_NOP*          = 0x00'u32
  OP_HALT*         = 0x01'u32
  OP_LOAD_F32*     = 0x10'u32
  OP_STORE_F32*    = 0x11'u32
  OP_LOAD_U8*      = 0x12'u32
  OP_LOAD_CONST*   = 0x13'u32
  OP_ADD*          = 0x20'u32
  OP_MUL*          = 0x21'u32
  OP_FMA*          = 0x22'u32
  OP_LUT_U8*       = 0x30'u32
  OP_FOR_BEGIN*    = 0x40'u32
  OP_FOR_END*      = 0x41'u32
  OP_BARRIER_TG*   = 0x50'u32  ## within one threadgroup only. Metal/Vulkan have NO
                                ## grid-wide barrier inside one dispatch — the queue
                                ## between two dispatches (sessionEncBarrier) is the
                                ## "device barrier". Merging fsc+ssm multi-TG = race.
  OP_TG_LOAD_F32*  = 0x60'u32
  OP_TG_STORE_F32* = 0x61'u32
  # ── new: enough ops to write SSM/RMSNorm/gemv 100% on the GPU, no CPU ──
  OP_SUB*          = 0x23'u32   # R[a] = R[b] - R[c]
  OP_DIV*          = 0x27'u32   # R[a] = R[b] / R[c]
  OP_EXP*          = 0x24'u32   # R[a] = exp(R[b])
  OP_NEG*          = 0x25'u32   # R[a] = -R[b]
  OP_SIGMOID*      = 0x26'u32   # R[a] = 1/(1+exp(-R[b]))
  OP_SOFTPLUS*     = 0x28'u32   # R[a] = log(1+exp(R[b]))
  OP_RSQRT*        = 0x29'u32   # R[a] = rsqrt(R[b])
  OP_LOAD_F32_IDX* = 0x14'u32   # R[a] = memF32[(uint)R[b]]           (DYNAMIC offset, not a constant)
  OP_STORE_F32_IDX* = 0x15'u32   # memF32[(uint)R[b]] = R[a]
  OP_ADD_CONST*    = 0x2A'u32   # R[a] = R[b] + const_f32 (next word)
  OP_MUL_CONST*    = 0x2B'u32   # R[a] = R[b] * const_f32 (next word)
  OP_CMP_LT*       = 0x70'u32   # R[a] = (R[b] < R[c]) ? 1.0 : 0.0
  OP_IF_BEGIN*     = 0x72'u32   # if (R[a] != 0.0f) {   -- like FOR_BEGIN/END, no absolute jump needed
  OP_IF_END*       = 0x73'u32   # }
  # ── new (GROUPED gated RMSNorm - see mamba2_vm_kernels.nim v2) ──
  OP_LOAD_LID*     = 0x16'u32   # R[a] = float(thread_position_in_threadgroup)
  OP_LOAD_TGID*    = 0x17'u32   # R[a] = float(threadgroup_position_in_grid)
  # ── new: INTEGER registers U[16] (uint32) for large addresses ─────────────
  # R[] is float32, so it represents integers exactly only up to 2^24 (16.7M);
  # the fp4 lm_head weights (32768 x 4096 / 2 = 67M bytes) are far above that, so
  # addresses MUST go through U[]. The U-* ops only run on the JIT path (jit_vm.nim);
  # the micro_vm.metal interpreter does not know them (unknown op -> halt).
  OP_U_TGID*       = 0x80'u32   # U[a] = threadgroup_position_in_grid
  OP_U_LID*        = 0x81'u32   # U[a] = thread_position_in_threadgroup
  OP_U_CONST*      = 0x82'u32   # U[a] = next word
  OP_U_ADD*        = 0x83'u32   # U[a] = U[b] + U[c]
  OP_U_MUL*        = 0x84'u32   # U[a] = U[b] * U[c]
  OP_U_ADDK*       = 0x85'u32   # U[a] = U[b] + word
  OP_U_MULK*       = 0x86'u32   # U[a] = U[b] * word
  OP_U_ANDK*       = 0x87'u32   # U[a] = U[b] & word
  OP_U_SHRK*       = 0x88'u32   # U[a] = U[b] >> word
  OP_U_SHLK*       = 0x89'u32   # U[a] = U[b] << word
  OP_U_SHR*        = 0x8A'u32   # U[a] = U[b] >> U[c]
  OP_U_LOAD_U8*    = 0x8B'u32   # U[a] = memU8[U[b] + word]           (buffer 2)
  OP_U_LOAD_CONSTF* = 0x8C'u32  # R[a] = constF32[U[b] + word]        (buffer 3)
  OP_U_LOAD_IN*    = 0x8D'u32   # R[a] = inF32[U[b] + word]           (buffer 4)
  OP_U_STORE_F32*  = 0x8E'u32   # memF32[U[b] + word] = R[a]          (buffer 1)
  OP_U_TO_R*       = 0x8F'u32   # R[a] = float(U[b])
  OP_R_TO_U*       = 0x90'u32   # U[a] = uint(R[b])
  OP_U_LOAD_F32*   = 0x91'u32   # R[a] = memF32[U[b] + word]          (buffer 1)
  OP_U_ATOMIC_ADD* = 0x92'u32   # atomic: memF32[U[b] + word] += R[a]  (buffer 1; MSL 3.0+ atomic_float)
  # ── ONE-DISPATCH (persistent kernel, many threadgroups, Metal JIT path only) ──────────
  # Metal has no grid-wide barrier inside one dispatch. OP_GRID_BARRIER is a SOFTWARE barrier
  # (atomic counter in memF32 + spin). SAFE ONLY IF every threadgroup of the dispatch is
  # resident at the same time (TG count <= GPU capacity) -> the host must pick a small ntg (see
  # onedispatch.nim, env BYBY_OD_TG) and run t_grid_barrier.nim to check the hardware first.
  OP_GRID_BARRIER* = 0x93'u32   # next word = float index of the ctrl block (4 uint: cnt,done,err,lastErr)
  OP_BASE_F32*     = 0x94'u32   # memF32 = memF32_goc + U[a]   (don vi float)
  OP_BASE_U8*      = 0x95'u32   # memU8  = memU8_goc  + U[a]   (don vi byte)
  OP_BASE_LUT*     = 0x96'u32   # lut    = lut_goc    + U[a]   (don vi float)
  OP_BASE_IN*      = 0x97'u32   # inF32  = inF32_goc  + U[a]   (don vi float)
  OP_U_LOAD_META*  = 0x98'u32   # U[a] = as_type<uint>(lut_goc[U[b] + word])  (a uint32 slot table hidden in buffer 3)
  OP_VTG_BEGIN*    = 0x99'u32   # a=mode (0 thuong, 1 sub32, 2 guard); word1=nVirt; word2=tham so (mode1: stride tgF32 float; mode2: NT)
  OP_VTG_END*      = 0x9A'u32
  OP_LOAD_VACT*    = 0x9B'u32   # R[a] = (this virtual threadgroup has real work) ? 1 : 0
  OP_U_LOAD_U32*   = 0x9D'u32   # U[a] = *(uint*)(memU8 + U[b] + word)  (4 bytes/lane, ADDRESS MUST be a multiple of 4; buffer 2)
  OP_U_LOAD_IN4*   = 0x9E'u32   # R[a..a+3] = inF32[U[b]+word .. +3]  (one 16-byte load; buffer 4)
  # ── simdgroup ops (Metal JIT and Vulkan subgroups; the test interpreter uses width 32) ──
  OP_SIMD_SUM*     = 0xA0'u32   # R[a] = sum of R[b] over the simdgroup (every lane must execute it)
  OP_SIMD_LANE*    = 0xA1'u32   # U[a] = thread_index_in_simdgroup
  OP_SIMD_ID*      = 0xA2'u32   # U[a] = simdgroup_index_in_threadgroup
  OP_SIMD_W*       = 0xA3'u32   # U[a] = threads_per_simdgroup
  OP_MM_ZERO*      = 0xA5'u32   # MM[a] = 8x8 float simdgroup matrix of zeros
  OP_MM_LOAD*      = 0xA6'u32   # MM[a] = simdgroup_load(tgF32 + U[b], ld = next word, transpose = c)
  OP_MM_MAC*       = 0xA7'u32   # MM[a] = MM[b] * MM[c] + MM[a]
  OP_MM_STORE_TG*  = 0xA8'u32   # simdgroup_store(MM[a], tgF32 + U[b], ld = next word)
  OP_MM_STORE_MEM* = 0xA9'u32   # simdgroup_store(MM[a], memF32 + U[b], ld = next word)
  OP_SIMD_GID*     = 0xA4'u32   # U[a] = global thread id / threads_per_simdgroup (grid-wide simdgroup index)
  OP_GRID_EPILOGUE* = 0x9C'u32  # next word = ctrl index; the last TG resets the counter, copies err -> lastErr

type
  VmProgram* = object
    words*: seq[uint32]

proc pack3(op, a, b, c: uint32): uint32 {.inline.} =
  ## 5-bit register fields: 32 float registers R[] and 32 int registers U[].
  ## (Was 4 bits = 16.) The hot FP4 GEMM needs 16 accumulators (8 rows x 2
  ## output columns) plus 4 vector-loaded inputs, 8 decoded weights, 2 group
  ## scales and a loop counter; 16 registers could not express that.
  (op and 0xFF'u32) or ((a and 0x1F) shl 8) or ((b and 0x1F) shl 13) or ((c and 0x1F) shl 18)

proc pack2(op, a, b: uint32): uint32 {.inline.} =
  (op and 0xFF'u32) or ((a and 0x1F) shl 8) or ((b and 0x1F) shl 13)

proc pack1(op, a: uint32): uint32 {.inline.} =
  (op and 0xFF'u32) or ((a and 0x1F) shl 8)

# ── Emit (writes a "file" in RAM) ──────────────────────────────

proc emit*(p: var VmProgram; w: uint32) =
  p.words.add w

proc emitNop*(p: var VmProgram) =
  p.emit OP_NOP

proc emitHalt*(p: var VmProgram) =
  p.emit OP_HALT

proc emitLoadF32*(p: var VmProgram; rd, rb: uint32; offset: uint32) =
  ## R[rd] = memF32[R[rb] + offset]
  p.emit pack2(OP_LOAD_F32, rd, rb)
  p.emit offset

proc emitStoreF32*(p: var VmProgram; rs, rb: uint32; offset: uint32) =
  p.emit pack2(OP_STORE_F32, rs, rb)
  p.emit offset

proc emitLoadU8*(p: var VmProgram; rd, rb: uint32; offset: uint32) =
  p.emit pack2(OP_LOAD_U8, rd, rb)
  p.emit offset

proc emitLoadConstF32*(p: var VmProgram; rd: uint32; v: float32) =
  p.emit pack1(OP_LOAD_CONST, rd)
  var bits: uint32
  copyMem(addr bits, unsafeAddr v, 4)
  p.emit bits

proc emitAdd*(p: var VmProgram; rd, ra, rb: uint32) =
  p.emit pack3(OP_ADD, rd, ra, rb)

proc emitMul*(p: var VmProgram; rd, ra, rb: uint32) =
  p.emit pack3(OP_MUL, rd, ra, rb)

proc emitFma*(p: var VmProgram; rd, ra, rb, rc: uint32) =
  p.emit pack3(OP_FMA, rd, ra, rb)
  p.emit (rc and 0x1F)

proc emitLutU8*(p: var VmProgram; rd, ri: uint32) =
  ## R[rd] = lut[uint(R[ri]) & 255]
  p.emit pack2(OP_LUT_U8, rd, ri)

proc emitForBegin*(p: var VmProgram; countReg: uint32) =
  ## GPU loop: remain = R[countReg]
  p.emit pack1(OP_FOR_BEGIN, countReg)

proc emitForEnd*(p: var VmProgram) =
  p.emit OP_FOR_END

proc emitBarrierTg*(p: var VmProgram) =
  p.emit OP_BARRIER_TG

proc emitTgLoadF32*(p: var VmProgram; rd, rb: uint32; offset: uint32) =
  p.emit pack2(OP_TG_LOAD_F32, rd, rb)
  p.emit offset

proc emitTgStoreF32*(p: var VmProgram; rs, rb: uint32; offset: uint32) =
  p.emit pack2(OP_TG_STORE_F32, rs, rb)
  p.emit offset

# ── emit for the new opcodes ───────────────────────────────────────

proc emitSub*(p: var VmProgram; rd, ra, rb: uint32) =
  p.emit pack3(OP_SUB, rd, ra, rb)

proc emitDiv*(p: var VmProgram; rd, ra, rb: uint32) =
  p.emit pack3(OP_DIV, rd, ra, rb)

proc emitExp*(p: var VmProgram; rd, rb: uint32) =
  p.emit pack2(OP_EXP, rd, rb)

proc emitNeg*(p: var VmProgram; rd, rb: uint32) =
  p.emit pack2(OP_NEG, rd, rb)

proc emitSigmoid*(p: var VmProgram; rd, rb: uint32) =
  p.emit pack2(OP_SIGMOID, rd, rb)

proc emitSoftplus*(p: var VmProgram; rd, rb: uint32) =
  p.emit pack2(OP_SOFTPLUS, rd, rb)

proc emitRsqrt*(p: var VmProgram; rd, rb: uint32) =
  p.emit pack2(OP_RSQRT, rd, rb)

proc emitLoadF32Idx*(p: var VmProgram; rd, rb: uint32) =
  ## R[rd] = memF32[(uint)R[rb]]  — DYNAMIC offset (computed at run time),
  ## unlike emitLoadF32 (STATIC offset fixed when the bytecode is built).
  p.emit pack2(OP_LOAD_F32_IDX, rd, rb)

proc emitStoreF32Idx*(p: var VmProgram; rs, rb: uint32) =
  p.emit pack2(OP_STORE_F32_IDX, rs, rb)

proc emitAddConst*(p: var VmProgram; rd, rb: uint32; v: float32) =
  p.emit pack2(OP_ADD_CONST, rd, rb)
  var bits: uint32
  copyMem(addr bits, unsafeAddr v, 4)
  p.emit bits

proc emitMulConst*(p: var VmProgram; rd, rb: uint32; v: float32) =
  p.emit pack2(OP_MUL_CONST, rd, rb)
  var bits: uint32
  copyMem(addr bits, unsafeAddr v, 4)
  p.emit bits

proc emitCmpLt*(p: var VmProgram; rd, ra, rb: uint32) =
  p.emit pack3(OP_CMP_LT, rd, ra, rb)

proc emitIfBegin*(p: var VmProgram; condReg: uint32) =
  p.emit pack1(OP_IF_BEGIN, condReg)

proc emitIfEnd*(p: var VmProgram) =
  p.emit OP_IF_END

proc emitLoadLid*(p: var VmProgram; rd: uint32) =
  p.emit pack1(OP_LOAD_LID, rd)

proc emitLoadTgid*(p: var VmProgram; rd: uint32) =
  p.emit pack1(OP_LOAD_TGID, rd)

proc opHasWord*(op: uint32): bool {.inline.} =
  ## Which ops read one extra operand word right after them (for skipBlock/disasm).
  case op
  of OP_LOAD_F32, OP_STORE_F32, OP_LOAD_U8, OP_LOAD_CONST, OP_FMA,
     OP_ADD_CONST, OP_MUL_CONST, OP_TG_LOAD_F32, OP_TG_STORE_F32,
     OP_U_CONST, OP_U_ADDK, OP_U_MULK, OP_U_ANDK, OP_U_SHRK, OP_U_SHLK,
     OP_U_LOAD_U8, OP_U_LOAD_U32, OP_U_LOAD_CONSTF, OP_U_LOAD_IN, OP_U_LOAD_IN4, OP_U_STORE_F32,
     OP_U_LOAD_F32, OP_U_ATOMIC_ADD, OP_MM_LOAD, OP_MM_STORE_TG, OP_MM_STORE_MEM: true
  else: false

proc emitSimdSum*(p: var VmProgram; rd, rs: uint32) = p.emit pack2(OP_SIMD_SUM, rd, rs)
proc emitSimdLane*(p: var VmProgram; ud: uint32) = p.emit pack1(OP_SIMD_LANE, ud)
proc emitSimdId*(p: var VmProgram; ud: uint32) = p.emit pack1(OP_SIMD_ID, ud)
proc emitMmZero*(p: var VmProgram; ma: uint32) = p.emit pack1(OP_MM_ZERO, ma)
proc emitMmLoad*(p: var VmProgram; ma, ub, transpose, ld: uint32) =
  p.emit pack3(OP_MM_LOAD, ma, ub, transpose); p.emit ld
proc emitMmMac*(p: var VmProgram; ma, mb, mc: uint32) = p.emit pack3(OP_MM_MAC, ma, mb, mc)
proc emitMmStoreTg*(p: var VmProgram; ma, ub, ld: uint32) =
  p.emit pack2(OP_MM_STORE_TG, ma, ub); p.emit ld
proc emitMmStoreMem*(p: var VmProgram; ma, ub, ld: uint32) =
  p.emit pack2(OP_MM_STORE_MEM, ma, ub); p.emit ld
proc emitSimdW*(p: var VmProgram; ud: uint32) = p.emit pack1(OP_SIMD_W, ud)
proc emitSimdGid*(p: var VmProgram; ud: uint32) = p.emit pack1(OP_SIMD_GID, ud)
proc emitUTgid*(p: var VmProgram; ud: uint32) = p.emit pack1(OP_U_TGID, ud)
proc emitULid*(p: var VmProgram; ud: uint32) = p.emit pack1(OP_U_LID, ud)
proc emitUConst*(p: var VmProgram; ud: uint32; v: uint32) =
  p.emit pack1(OP_U_CONST, ud); p.emit v
proc emitUAdd*(p: var VmProgram; ud, ua, ub: uint32) = p.emit pack3(OP_U_ADD, ud, ua, ub)
proc emitUMul*(p: var VmProgram; ud, ua, ub: uint32) = p.emit pack3(OP_U_MUL, ud, ua, ub)
proc emitUShr*(p: var VmProgram; ud, ua, ub: uint32) = p.emit pack3(OP_U_SHR, ud, ua, ub)
proc emitUAddK*(p: var VmProgram; ud, ua: uint32; k: uint32) =
  p.emit pack2(OP_U_ADDK, ud, ua); p.emit k
proc emitUMulK*(p: var VmProgram; ud, ua: uint32; k: uint32) =
  p.emit pack2(OP_U_MULK, ud, ua); p.emit k
proc emitUAndK*(p: var VmProgram; ud, ua: uint32; k: uint32) =
  p.emit pack2(OP_U_ANDK, ud, ua); p.emit k
proc emitUShrK*(p: var VmProgram; ud, ua: uint32; k: uint32) =
  p.emit pack2(OP_U_SHRK, ud, ua); p.emit k
proc emitUShlK*(p: var VmProgram; ud, ua: uint32; k: uint32) =
  p.emit pack2(OP_U_SHLK, ud, ua); p.emit k
proc emitULoadU8*(p: var VmProgram; ud, ub: uint32; off: uint32 = 0) =
  p.emit pack2(OP_U_LOAD_U8, ud, ub); p.emit off
proc emitULoadU32*(p: var VmProgram; ud, ub: uint32; off: uint32 = 0) =
  ## Loads 32 bits (4 consecutive bytes, little-endian) from memU8[U[ub]+off]. The address MUST be a multiple of 4.
  p.emit pack2(OP_U_LOAD_U32, ud, ub); p.emit off

proc emitULoadConstF*(p: var VmProgram; rd, ub: uint32; off: uint32 = 0) =
  p.emit pack2(OP_U_LOAD_CONSTF, rd, ub); p.emit off
proc emitULoadIn*(p: var VmProgram; rd, ub: uint32; off: uint32 = 0) =
  p.emit pack2(OP_U_LOAD_IN, rd, ub); p.emit off
proc emitULoadIn4*(p: var VmProgram; rd, ub: uint32; off: uint32 = 0) =
  ## R[rd..rd+3] = 4 consecutive floats of inF32 at U[ub]+off. The address must be
  ## a multiple of 4 floats (16 bytes) - the kernel must guarantee it.
  doAssert rd + 3 <= 31, "emitULoadIn4: exceeds the 32 R[] registers"
  p.emit pack2(OP_U_LOAD_IN4, rd, ub); p.emit off
proc emitUStoreF32*(p: var VmProgram; rs, ub: uint32; off: uint32 = 0) =
  p.emit pack2(OP_U_STORE_F32, rs, ub); p.emit off
proc emitULoadF32*(p: var VmProgram; rd, ub: uint32; off: uint32 = 0) =
  p.emit pack2(OP_U_LOAD_F32, rd, ub); p.emit off
proc emitUAtomicAdd*(p: var VmProgram; rs, ub: uint32; off: uint32 = 0) =
  ## memF32[U[ub] + off] += R[rs]  (atomic, for gradients shared between threads)
  p.emit pack2(OP_U_ATOMIC_ADD, rs, ub); p.emit off
proc emitUToR*(p: var VmProgram; rd, ub: uint32) = p.emit pack2(OP_U_TO_R, rd, ub)
proc emitRToU*(p: var VmProgram; ud, rb: uint32) = p.emit pack2(OP_R_TO_U, ud, rb)

template sessionVmIfGpu*(p: var VmProgram; condReg: uint32; body: untyped) =
  ## if (R[condReg] != 0) on the GPU — writes IF_BEGIN/body/IF_END into the bytecode.
  p.emitIfBegin(condReg)
  body
  p.emitIfEnd()

# ── sessionFor = a HOST for loop while writing the program ────────────────────

template sessionVmFor*(n: int; body: untyped) =
  ## Host path: loops while emitting (unrolls the program). Not a GPU loop.
  for vmIdx in 0 ..< n:
    body

template sessionVmForGpu*(p: var VmProgram; countReg: uint32; body: untyped) =
  ## GPU path: writes FOR_BEGIN / body / FOR_END into the bytecode.
  p.emitForBegin(countReg)
  body
  p.emitForEnd()

# ── ONE-DISPATCH emit helpers ───────────────────────────────

proc emitGridBarrier*(p: var VmProgram; ctrlIdx: uint32) =
  p.emit OP_GRID_BARRIER; p.emit ctrlIdx
proc emitGridEpilogue*(p: var VmProgram; ctrlIdx: uint32) =
  p.emit OP_GRID_EPILOGUE; p.emit ctrlIdx
proc emitBaseF32*(p: var VmProgram; ua: uint32) = p.emit pack1(OP_BASE_F32, ua)
proc emitBaseU8*(p: var VmProgram; ua: uint32) = p.emit pack1(OP_BASE_U8, ua)
proc emitBaseLut*(p: var VmProgram; ua: uint32) = p.emit pack1(OP_BASE_LUT, ua)
proc emitBaseIn*(p: var VmProgram; ua: uint32) = p.emit pack1(OP_BASE_IN, ua)
proc emitULoadMeta*(p: var VmProgram; ud, ub: uint32; off: uint32 = 0) =
  p.emit pack2(OP_U_LOAD_META, ud, ub); p.emit off
proc emitVtgBegin*(p: var VmProgram; mode: uint32; nVirt, param: uint32) =
  p.emit pack1(OP_VTG_BEGIN, mode); p.emit nVirt; p.emit param
proc emitVtgEnd*(p: var VmProgram) = p.emit OP_VTG_END
proc emitLoadVact*(p: var VmProgram; rd: uint32) = p.emit pack1(OP_LOAD_VACT, rd)

# ── Save / load .bvm files ─────────────────────────────────────

proc saveBvm*(p: VmProgram; path: string) =
  ## File = raw uint32 LE words.
  var f = newFileStream(path, fmWrite)
  if f.isNil: raise newException(IOError, "cannot write " & path)
  defer: f.close()
  for w in p.words:
    var le = w
    f.writeData(addr le, 4)

proc loadBvm*(path: string): VmProgram =
  var f = newFileStream(path, fmRead)
  if f.isNil: raise newException(IOError, "cannot read " & path)
  defer: f.close()
  let data = f.readAll()
  let n = data.len div 4
  result.words = newSeq[uint32](n)
  if n > 0:
    copyMem(addr result.words[0], unsafeAddr data[0], n * 4)

# ── Example: a "kernel" decoding one fp4-ish byte + fma (illustration) ───

proc buildExampleLutFma*(): VmProgram =
  ## R0 = tid (set by the kernel)
  ## Assume: memU8[tid] = index, lut[] ready, memF32[tid] *= scale then fma
  var p: VmProgram
  p.emitLoadU8(1, 0, 0)          # R1 = float(memU8[tid])
  p.emitLutU8(2, 1)              # R2 = lut[R1]
  p.emitLoadF32(3, 0, 0)         # R3 = memF32[tid]  (activation / acc)
  p.emitLoadConstF32(4, 1.0'f32) # R4 = 1
  p.emitFma(3, 2, 4, 3)          # R3 = R2*1 + R3
  p.emitStoreF32(3, 0, 0)        # memF32[tid] = R3
  p.emitHalt()
  result = p

# ── Running (needs metal_enc_* / a session pipeline) ────────────
#
# Pseudocode (with metal_enc_pipeline / bind / dispatch available):
#
#   sessionBegin(be)
#   hCode = sessionUploadU32(p.words)   # or upload bytes
#   hF32  = ...
#   hU8   = ...
#   hLut  = ...
#   sessionPipeline(be, "micro_vm_kernel")
#   sessionBind(0, hCode)
#   sessionBind(1, hF32)
#   sessionBind(2, hU8)
#   sessionBind(3, hLut)
#   sessionBindInt(4, p.words.len)
#   sessionBindInt(5, tgFloats)
#   sessionDispatch(grid, tg)
#   sessionEnd(be)
#
# Uploading uint32 / binding the threadgroup length depends on the current shim;
# without sessionUploadU32, upload bytes and cast.

when isMainModule:
  var ex = buildExampleLutFma()
  echo "example words: ", ex.words.len
  ex.saveBvm("example.bvm")
  echo "wrote example.bvm"
