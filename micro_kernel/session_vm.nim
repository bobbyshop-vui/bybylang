## session_vm.nim — Cách A: ghi bytecode (file/program) rồi chạy bằng micro_vm_kernel
##
## Không ghép sessionMatmul cấp cao. Dev “viết kernel mới” = emit op siêu thấp.
##
## sessionFor = for trên HOST lúc ghi program (hoặc dùng OP_FOR_BEGIN/END trong bytecode).

import std/streams

# ── Opcode (khớp micro_vm.metal) ─────────────────────────────
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
  OP_BARRIER_TG*   = 0x50'u32  ## chi trong 1 threadgroup. Metal/Vulkan KHONG co
                                ## barrier toan grid trong 1 dispatch — hang doi
                                ## giua 2 dispatch (sessionEncBarrier) moi la
                                ## "device barrier". Gop fsc+ssm multi-TG = race.
  OP_TG_LOAD_F32*  = 0x60'u32
  OP_TG_STORE_F32* = 0x61'u32
  # ── mới: đủ phép để viết SSM/RMSNorm/gemv 100% GPU, không CPU ──
  OP_SUB*          = 0x23'u32   # R[a] = R[b] - R[c]
  OP_DIV*          = 0x27'u32   # R[a] = R[b] / R[c]
  OP_EXP*          = 0x24'u32   # R[a] = exp(R[b])
  OP_NEG*          = 0x25'u32   # R[a] = -R[b]
  OP_SIGMOID*      = 0x26'u32   # R[a] = 1/(1+exp(-R[b]))
  OP_SOFTPLUS*     = 0x28'u32   # R[a] = log(1+exp(R[b]))
  OP_RSQRT*        = 0x29'u32   # R[a] = rsqrt(R[b])
  OP_LOAD_F32_IDX* = 0x14'u32   # R[a] = memF32[(uint)R[b]]           (offset ĐỘNG, không phải const)
  OP_STORE_F32_IDX* = 0x15'u32   # memF32[(uint)R[b]] = R[a]
  OP_ADD_CONST*    = 0x2A'u32   # R[a] = R[b] + const_f32 (word kế tiếp)
  OP_MUL_CONST*    = 0x2B'u32   # R[a] = R[b] * const_f32 (word kế tiếp)
  OP_CMP_LT*       = 0x70'u32   # R[a] = (R[b] < R[c]) ? 1.0 : 0.0
  OP_IF_BEGIN*     = 0x72'u32   # if (R[a] != 0.0f) {   -- giống FOR_BEGIN/END, không cần jump tuyệt đối
  OP_IF_END*       = 0x73'u32   # }
  # ── mới (gated RMSNorm THEO NHÓM - xem mamba2_vm_kernels.nim v2) ──
  OP_LOAD_LID*     = 0x16'u32   # R[a] = float(thread_position_in_threadgroup)
  OP_LOAD_TGID*    = 0x17'u32   # R[a] = float(threadgroup_position_in_grid)
  # ── mới: thanh ghi SỐ NGUYÊN U[16] (uint32) cho địa chỉ lớn ─────────────
  # R[] là float32 nên chỉ biểu diễn chính xác số nguyên <= 2^24 (16.7M);
  # weight fp4 của lm_head (32768 x 4096 / 2 = 67M byte) vượt xa mức đó nên
  # địa chỉ PHẢI đi qua U[]. Các op U-* chỉ chạy trên đường JIT (jit_vm.nim);
  # interpreter micro_vm.metal không hiểu chúng (gặp op lạ -> halt).
  OP_U_TGID*       = 0x80'u32   # U[a] = threadgroup_position_in_grid
  OP_U_LID*        = 0x81'u32   # U[a] = thread_position_in_threadgroup
  OP_U_CONST*      = 0x82'u32   # U[a] = word kế tiếp
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
  # ── ONE-DISPATCH (persistent kernel, nhieu threadgroup, chi duong JIT Metal) ──────────
  # Metal khong co barrier toan grid trong 1 dispatch. OP_GRID_BARRIER la barrier PHAN MEM
  # (atomic counter trong memF32 + spin). AN TOAN CHI KHI moi threadgroup cua dispatch deu
  # dang resident cung luc (so TG <= dung luong GPU) -> host phai chon ntg nho (xem
  # onedispatch.nim, env BYBY_OD_TG) va chay t_grid_barrier.nim de kiem tra phan cung truoc.
  OP_GRID_BARRIER* = 0x93'u32   # word kế tiếp = chi so float cua khoi ctrl (4 uint: cnt,done,err,lastErr)
  OP_BASE_F32*     = 0x94'u32   # memF32 = memF32_goc + U[a]   (don vi float)
  OP_BASE_U8*      = 0x95'u32   # memU8  = memU8_goc  + U[a]   (don vi byte)
  OP_BASE_LUT*     = 0x96'u32   # lut    = lut_goc    + U[a]   (don vi float)
  OP_BASE_IN*      = 0x97'u32   # inF32  = inF32_goc  + U[a]   (don vi float)
  OP_U_LOAD_META*  = 0x98'u32   # U[a] = as_type<uint>(lut_goc[U[b] + word])  (bang slot uint32 giau trong buffer 3)
  OP_VTG_BEGIN*    = 0x99'u32   # a=mode (0 thuong, 1 sub32, 2 guard); word1=nVirt; word2=tham so (mode1: stride tgF32 float; mode2: NT)
  OP_VTG_END*      = 0x9A'u32
  OP_LOAD_VACT*    = 0x9B'u32   # R[a] = (threadgroup ao nay co viec that) ? 1 : 0
  OP_U_LOAD_U32*   = 0x9D'u32   # U[a] = *(uint*)(memU8 + U[b] + word)  (nap 4 byte/lane, DIA CHI PHAI chia het 4; buffer 2)
  OP_GRID_EPILOGUE* = 0x9C'u32  # word kế tiếp = chi so ctrl; TG cuoi cung reset counter, chep err -> lastErr

type
  VmProgram* = object
    words*: seq[uint32]

proc pack3(op, a, b, c: uint32): uint32 {.inline.} =
  (op and 0xFF'u32) or ((a and 0xF) shl 8) or ((b and 0xF) shl 12) or ((c and 0xF) shl 16)

proc pack2(op, a, b: uint32): uint32 {.inline.} =
  (op and 0xFF'u32) or ((a and 0xF) shl 8) or ((b and 0xF) shl 12)

proc pack1(op, a: uint32): uint32 {.inline.} =
  (op and 0xFF'u32) or ((a and 0xF) shl 8)

# ── Emit (ghi “file” trong RAM) ──────────────────────────────

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
  p.emit (rc and 0xF)

proc emitLutU8*(p: var VmProgram; rd, ri: uint32) =
  ## R[rd] = lut[uint(R[ri]) & 255]
  p.emit pack2(OP_LUT_U8, rd, ri)

proc emitForBegin*(p: var VmProgram; countReg: uint32) =
  ## Vòng trên GPU: remain = R[countReg]
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

# ── emit cho opcode mới ───────────────────────────────────────

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
  ## R[rd] = memF32[(uint)R[rb]]  — offset ĐỘNG (chạy thời gian thực),
  ## khác emitLoadF32 (offset TĨNH ghi cứng lúc build bytecode).
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
  ## Op nào đọc thêm 1 word toán hạng ngay sau nó (dùng cho skipBlock/disasm).
  case op
  of OP_LOAD_F32, OP_STORE_F32, OP_LOAD_U8, OP_LOAD_CONST, OP_FMA,
     OP_ADD_CONST, OP_MUL_CONST, OP_TG_LOAD_F32, OP_TG_STORE_F32,
     OP_U_CONST, OP_U_ADDK, OP_U_MULK, OP_U_ANDK, OP_U_SHRK, OP_U_SHLK,
     OP_U_LOAD_U8, OP_U_LOAD_U32, OP_U_LOAD_CONSTF, OP_U_LOAD_IN, OP_U_STORE_F32,
     OP_U_LOAD_F32, OP_U_ATOMIC_ADD: true
  else: false

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
  ## Nap 32-bit (4 byte lien tiep, little-endian) tu memU8[U[ub]+off]. Dia chi PHAI chia het 4.
  p.emit pack2(OP_U_LOAD_U32, ud, ub); p.emit off

proc emitULoadConstF*(p: var VmProgram; rd, ub: uint32; off: uint32 = 0) =
  p.emit pack2(OP_U_LOAD_CONSTF, rd, ub); p.emit off
proc emitULoadIn*(p: var VmProgram; rd, ub: uint32; off: uint32 = 0) =
  p.emit pack2(OP_U_LOAD_IN, rd, ub); p.emit off
proc emitUStoreF32*(p: var VmProgram; rs, ub: uint32; off: uint32 = 0) =
  p.emit pack2(OP_U_STORE_F32, rs, ub); p.emit off
proc emitULoadF32*(p: var VmProgram; rd, ub: uint32; off: uint32 = 0) =
  p.emit pack2(OP_U_LOAD_F32, rd, ub); p.emit off
proc emitUAtomicAdd*(p: var VmProgram; rs, ub: uint32; off: uint32 = 0) =
  ## memF32[U[ub] + off] += R[rs]  (atomic, dùng cho gradient dùng chung giữa các thread)
  p.emit pack2(OP_U_ATOMIC_ADD, rs, ub); p.emit off
proc emitUToR*(p: var VmProgram; rd, ub: uint32) = p.emit pack2(OP_U_TO_R, rd, ub)
proc emitRToU*(p: var VmProgram; ud, rb: uint32) = p.emit pack2(OP_R_TO_U, ud, rb)

template sessionVmIfGpu*(p: var VmProgram; condReg: uint32; body: untyped) =
  ## if (R[condReg] != 0) trên GPU — ghi IF_BEGIN/body/IF_END vào bytecode.
  p.emitIfBegin(condReg)
  body
  p.emitIfEnd()

# ── sessionFor = for HOST lúc ghi program ────────────────────

template sessionVmFor*(n: int; body: untyped) =
  ## Đường host: lặp lúc emit (unroll program). Không phải for GPU.
  for vmIdx in 0 ..< n:
    body

template sessionVmForGpu*(p: var VmProgram; countReg: uint32; body: untyped) =
  ## Đường GPU: ghi FOR_BEGIN / body / FOR_END vào bytecode.
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

# ── Lưu / load file .bvm ─────────────────────────────────────

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

# ── Ví dụ: “kernel” decode 1 byte fp4-ish + fma (minh họa) ───

proc buildExampleLutFma*(): VmProgram =
  ## R0 = tid (kernel set sẵn)
  ## Giả sử: memU8[tid] = index, lut[] sẵn, memF32[tid] *= scale rồi fma
  var p: VmProgram
  p.emitLoadU8(1, 0, 0)          # R1 = float(memU8[tid])
  p.emitLutU8(2, 1)              # R2 = lut[R1]
  p.emitLoadF32(3, 0, 0)         # R3 = memF32[tid]  (activation / acc)
  p.emitLoadConstF32(4, 1.0'f32) # R4 = 1
  p.emitFma(3, 2, 4, 3)          # R3 = R2*1 + R3
  p.emitStoreF32(3, 0, 0)        # memF32[tid] = R3
  p.emitHalt()
  result = p

# ── Chạy (cần nối metal_enc_* / session pipeline) ────────────
#
# Pseudocode (bạn đã có metal_enc_pipeline / bind / dispatch):
#
#   sessionBegin(be)
#   hCode = sessionUploadU32(p.words)   # hoặc upload bytes
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
# Phần upload uint32 / bind threadgroup length phụ thuộc shim hiện tại;
# nếu chưa có sessionUploadU32, upload dưới dạng bytes rồi cast.

when isMainModule:
  var ex = buildExampleLutFma()
  echo "example words: ", ex.words.len
  ex.saveBvm("example.bvm")
  echo "wrote example.bvm"
