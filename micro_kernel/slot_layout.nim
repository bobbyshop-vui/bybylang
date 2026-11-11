## Validate recorded bytecode and conservatively classify its memory domains.
import std/[sets, strutils]
import ./session_vm

type VmAccess* = object
  reads*, writes*: HashSet[string]
  virtual*, grid*: bool
  atomic*: bool

proc vmAccess*(words: openArray[uint32]): VmAccess =
  var pc = 0
  var scopes: seq[uint32]
  var halted = false
  while pc < words.len:
    let op = words[pc] and 255
    inc pc
    var extra = 0
    if op in [OP_VTG_BEGIN, OP_VTG_END, OP_LOAD_VACT]: result.virtual = true
    if op in [OP_GRID_BARRIER, OP_GRID_EPILOGUE]: result.grid = true
    if op == OP_U_ATOMIC_ADD: result.atomic = true
    case op
    of OP_LOAD_F32, OP_U_LOAD_F32, OP_LOAD_F32_IDX:
      result.reads.incl "mem"
    of OP_STORE_F32, OP_U_STORE_F32, OP_STORE_F32_IDX:
      result.writes.incl "mem"
    of OP_U_ATOMIC_ADD:
      result.reads.incl "mem"; result.writes.incl "mem"
    of OP_LOAD_U8, OP_U_LOAD_U8, OP_U_LOAD_U32: result.reads.incl "u8"
    of OP_LUT_U8, OP_U_LOAD_CONSTF, OP_U_LOAD_META: result.reads.incl "lut"
    of OP_U_LOAD_IN, OP_U_LOAD_IN4: result.reads.incl "input"
    of OP_GRID_BARRIER, OP_GRID_EPILOGUE:
      # Their control words refer to a device-side synchronization buffer.
      result.reads.incl "u32"; result.writes.incl "u32"
    else: discard
    case op
    of OP_LOAD_F32, OP_STORE_F32, OP_LOAD_U8, OP_LOAD_CONST, OP_FMA,
       OP_ADD_CONST, OP_MUL_CONST, OP_TG_LOAD_F32, OP_TG_STORE_F32,
       OP_U_CONST, OP_U_ADDK, OP_U_MULK, OP_U_ANDK, OP_U_SHRK, OP_U_SHLK,
       OP_U_LOAD_U8, OP_U_LOAD_U32, OP_U_LOAD_CONSTF, OP_U_LOAD_IN,
       OP_U_LOAD_IN4, OP_U_STORE_F32, OP_U_LOAD_F32, OP_U_ATOMIC_ADD,
       OP_U_LOAD_META, OP_GRID_BARRIER, OP_GRID_EPILOGUE: extra = 1
    of OP_VTG_BEGIN: extra = 2
    of OP_NOP, OP_HALT, OP_LOAD_F32_IDX, OP_STORE_F32_IDX, OP_LOAD_LID,
       OP_LOAD_TGID, OP_ADD, OP_MUL, OP_SUB, OP_EXP, OP_NEG, OP_SIGMOID,
       OP_DIV, OP_SOFTPLUS, OP_RSQRT, OP_LUT_U8, OP_FOR_BEGIN, OP_FOR_END,
       OP_BARRIER_TG, OP_CMP_LT, OP_IF_BEGIN, OP_IF_END, OP_U_TGID, OP_U_LID,
       OP_U_ADD, OP_U_MUL, OP_U_SHR, OP_U_TO_R, OP_R_TO_U, OP_BASE_F32,
       OP_BASE_U8, OP_BASE_LUT, OP_BASE_IN, OP_VTG_END, OP_LOAD_VACT,
       OP_SIMD_SUM, OP_SIMD_LANE, OP_SIMD_ID, OP_SIMD_W, OP_SIMD_GID: discard
    else: raise newException(ValueError, "unknown VM opcode 0x" & toHex(op, 2))
    if pc+extra > words.len: raise newException(ValueError, "truncated recorded VM instruction")
    case op
    of OP_FOR_BEGIN, OP_IF_BEGIN, OP_VTG_BEGIN: scopes.add op
    of OP_FOR_END, OP_IF_END, OP_VTG_END:
      let opening = if op == OP_FOR_END: OP_FOR_BEGIN elif op == OP_IF_END: OP_IF_BEGIN else: OP_VTG_BEGIN
      if scopes.len == 0 or scopes[^1] != opening:
        raise newException(ValueError, "unbalanced recorded VM scope")
      scopes.setLen(scopes.len-1)
    of OP_HALT:
      if pc != words.len: raise newException(ValueError, "recorded VM halt before end")
      halted = true
    else: discard
    pc += extra
  if scopes.len > 0 or not halted: raise newException(ValueError, "recorded VM missing end scope or halt")
