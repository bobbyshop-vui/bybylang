// micro_vm.metal — full opcode interpreter (Cách A)
// Hỗ trợ đủ opcode cho Mamba2 SSM / conv1d / RMS+gate 100% GPU.
// Compile vào cùng library với vecop_matmul.metal (hoặc library riêng rồi link).
//
// Mỗi thread: private R[16], loop/if stack; đọc bytecode; thực thi op siêu thấp.
// Threadgroup memory: tgF32[] dùng chung trong 1 dispatch.

#include <metal_stdlib>
using namespace metal;

constant uint OP_NOP           = 0x00u;
constant uint OP_HALT          = 0x01u;
constant uint OP_LOAD_F32      = 0x10u;
constant uint OP_STORE_F32     = 0x11u;
constant uint OP_LOAD_U8       = 0x12u;
constant uint OP_LOAD_CONST    = 0x13u;
constant uint OP_LOAD_F32_IDX  = 0x14u;
constant uint OP_STORE_F32_IDX = 0x15u;
constant uint OP_LOAD_LID      = 0x16u;   // R[a] = thread_position_in_threadgroup
constant uint OP_LOAD_TGID     = 0x17u;   // R[a] = threadgroup_position_in_grid
constant uint OP_ADD           = 0x20u;
constant uint OP_MUL           = 0x21u;
constant uint OP_FMA           = 0x22u;
constant uint OP_SUB           = 0x23u;
constant uint OP_EXP           = 0x24u;
constant uint OP_NEG           = 0x25u;
constant uint OP_SIGMOID       = 0x26u;
constant uint OP_DIV           = 0x27u;
constant uint OP_SOFTPLUS      = 0x28u;
constant uint OP_RSQRT         = 0x29u;
constant uint OP_ADD_CONST     = 0x2Au;
constant uint OP_MUL_CONST     = 0x2Bu;
constant uint OP_LUT_U8        = 0x30u;
constant uint OP_FOR_BEGIN     = 0x40u;
constant uint OP_FOR_END       = 0x41u;
constant uint OP_BARRIER_TG    = 0x50u;
constant uint OP_TG_LOAD_F32   = 0x60u;
constant uint OP_TG_STORE_F32  = 0x61u;
constant uint OP_CMP_LT        = 0x70u;
constant uint OP_IF_BEGIN      = 0x72u;
constant uint OP_IF_END        = 0x73u;

inline uint opCode(uint w)  { return w & 0xFFu; }
inline uint opA(uint w)     { return (w >> 8)  & 0xFu; }
inline uint opB(uint w)     { return (w >> 12) & 0xFu; }
inline uint opC(uint w)     { return (w >> 16) & 0xFu; }

struct LoopFrame {
  uint backPC;
  int  remain;
};

// Helper: skip nested FOR/IF blocks (used when remain<=0 or condition false)
static inline void skipBlock(device const uint* code, thread uint& pc, uint nWords, uint endOp) {
  int depth = 1;
  while (pc < nWords && depth > 0) {
    uint w2 = code[pc++];
    uint o2 = opCode(w2);
    if (o2 == OP_FOR_BEGIN || o2 == OP_IF_BEGIN) ++depth;
    else if (o2 == OP_FOR_END || o2 == OP_IF_END) --depth;
    else if (o2 == OP_LOAD_F32 || o2 == OP_STORE_F32 || o2 == OP_LOAD_U8 ||
             o2 == OP_LOAD_CONST || o2 == OP_FMA ||
             o2 == OP_ADD_CONST || o2 == OP_MUL_CONST ||
             o2 == OP_TG_LOAD_F32 || o2 == OP_TG_STORE_F32) {
      if (pc < nWords) ++pc; // skip operand word
    }
  }
}

kernel void micro_vm_kernel(
    device const uint*  code      [[buffer(0)]],
    device float*       memF32    [[buffer(1)]],
    device const uchar* memU8     [[buffer(2)]],
    device const float* lut       [[buffer(3)]],
    constant uint&      nWords    [[buffer(4)]],
    constant uint&      tgFloats  [[buffer(5)]],
    threadgroup float*  tgF32     [[threadgroup(0)]],
    uint tid                      [[thread_position_in_grid]],
    uint lid                      [[thread_position_in_threadgroup]],
    uint tgid                     [[threadgroup_position_in_grid]],
    uint tgSize                   [[threads_per_threadgroup]])
{
  float R[16];
  for (int i = 0; i < 16; ++i) R[i] = 0.0f;
  R[0] = float(tid);

  LoopFrame stack[8];
  int sp = 0;
  uint pc = 0u;

  const uint kMaxSteps = 1u << 16;
  uint steps = 0u;

  while (pc < nWords && steps < kMaxSteps) {
    ++steps;
    uint w = code[pc++];
    uint op = opCode(w);

    switch (op) {
      case OP_NOP: continue;
      case OP_HALT: return;

      case OP_LOAD_F32: {
        uint rd = opA(w), rb = opB(w);
        if (pc >= nWords) return;
        uint off = code[pc++];
        R[rd] = memF32[uint(R[rb]) + off];
        continue;
      }
      case OP_STORE_F32: {
        uint rs = opA(w), rb = opB(w);
        if (pc >= nWords) return;
        uint off = code[pc++];
        memF32[uint(R[rb]) + off] = R[rs];
        continue;
      }
      case OP_LOAD_U8: {
        uint rd = opA(w), rb = opB(w);
        if (pc >= nWords) return;
        uint off = code[pc++];
        R[rd] = float(memU8[uint(R[rb]) + off]);
        continue;
      }
      case OP_LOAD_CONST: {
        uint rd = opA(w);
        if (pc >= nWords) return;
        uint bits = code[pc++];
        R[rd] = as_type<float>(bits);
        continue;
      }
      case OP_LOAD_F32_IDX: {
        // R[a] = memF32[(uint)R[b]]
        R[opA(w)] = memF32[uint(R[opB(w)])];
        continue;
      }
      case OP_STORE_F32_IDX: {
        // memF32[(uint)R[b]] = R[a]
        memF32[uint(R[opB(w)])] = R[opA(w)];
        continue;
      }
      case OP_LOAD_LID:  { R[opA(w)] = float(lid);  continue; }
      case OP_LOAD_TGID: { R[opA(w)] = float(tgid); continue; }

      case OP_ADD:   R[opA(w)] = R[opB(w)] + R[opC(w)]; continue;
      case OP_MUL:   R[opA(w)] = R[opB(w)] * R[opC(w)]; continue;
      case OP_SUB:   R[opA(w)] = R[opB(w)] - R[opC(w)]; continue;
      case OP_DIV:   R[opA(w)] = R[opB(w)] / R[opC(w)]; continue;
      case OP_FMA: {
        uint rd = opA(w), ra = opB(w), rb = opC(w);
        if (pc >= nWords) return;
        uint rc = code[pc++] & 0xFu;
        R[rd] = fma(R[ra], R[rb], R[rc]);
        continue;
      }
      case OP_EXP:      R[opA(w)] = exp(R[opB(w)]); continue;
      case OP_NEG:      R[opA(w)] = -R[opB(w)]; continue;
      case OP_SIGMOID:  R[opA(w)] = 1.0f / (1.0f + exp(-R[opB(w)])); continue;
      case OP_SOFTPLUS: R[opA(w)] = log(1.0f + exp(R[opB(w)])); continue;
      case OP_RSQRT:    R[opA(w)] = rsqrt(R[opB(w)]); continue;

      case OP_ADD_CONST: {
        uint rd = opA(w), rb = opB(w);
        if (pc >= nWords) return;
        float c = as_type<float>(code[pc++]);
        R[rd] = R[rb] + c;
        continue;
      }
      case OP_MUL_CONST: {
        uint rd = opA(w), rb = opB(w);
        if (pc >= nWords) return;
        float c = as_type<float>(code[pc++]);
        R[rd] = R[rb] * c;
        continue;
      }

      case OP_LUT_U8: {
        uint rd = opA(w), ri = opB(w);
        R[rd] = lut[uint(R[ri]) & 255u];
        continue;
      }

      case OP_FOR_BEGIN: {
        uint countReg = opA(w);
        int remain = int(R[countReg]);
        if (remain <= 0) {
          skipBlock(code, pc, nWords, OP_FOR_END);
          continue;
        }
        if (sp >= 8) return;
        stack[sp].backPC = pc;
        stack[sp].remain = remain - 1;
        ++sp;
        continue;
      }
      case OP_FOR_END: {
        if (sp <= 0) return;
        LoopFrame f = stack[sp - 1];
        if (f.remain > 0) {
          stack[sp - 1].remain = f.remain - 1;
          pc = f.backPC;
        } else {
          --sp;
        }
        continue;
      }

      case OP_IF_BEGIN: {
        // if (R[a] != 0) { body } else skip to matching IF_END
        if (R[opA(w)] == 0.0f) {
          skipBlock(code, pc, nWords, OP_IF_END);
        }
        // else fall through (body executes)
        continue;
      }
      case OP_IF_END:
        continue;

      case OP_BARRIER_TG: {
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        continue;
      }
      case OP_TG_LOAD_F32: {
        uint rd = opA(w), rb = opB(w);
        if (pc >= nWords) return;
        uint off = code[pc++];
        uint idx = uint(R[rb]) + off;
        if (idx < tgFloats) R[rd] = tgF32[idx];
        continue;
      }
      case OP_TG_STORE_F32: {
        uint rs = opA(w), rb = opB(w);
        if (pc >= nWords) return;
        uint off = code[pc++];
        uint idx = uint(R[rb]) + off;
        if (idx < tgFloats) tgF32[idx] = R[rs];
        continue;
      }

      case OP_CMP_LT:
        R[opA(w)] = (R[opB(w)] < R[opC(w)]) ? 1.0f : 0.0f;
        continue;

      default:
        // unknown op → halt
        return;
    }
  }
}
