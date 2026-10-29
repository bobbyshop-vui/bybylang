/* Nimformers CPU micro-VM interpreter.
 *
 * Executes the same packed bytecode as the Metal interpreter / Vulkan JIT on
 * the host.  Threads advance in lockstep so OP_BARRIER_TG and OP_GRID_BARRIER
 * remain meaningful without GPU hardware.
 */
#include "cpu_kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VM_MAX_THREADS 65536
#define VM_MAX_LOOPS 8
#define VM_MAX_STEPS (1u << 24)

enum {
    OP_NOP = 0x00, OP_HALT = 0x01,
    OP_LOAD_F32 = 0x10, OP_STORE_F32 = 0x11, OP_LOAD_U8 = 0x12,
    OP_LOAD_CONST = 0x13, OP_LOAD_F32_IDX = 0x14, OP_STORE_F32_IDX = 0x15,
    OP_LOAD_LID = 0x16, OP_LOAD_TGID = 0x17,
    OP_ADD = 0x20, OP_MUL = 0x21, OP_FMA = 0x22, OP_SUB = 0x23,
    OP_EXP = 0x24, OP_NEG = 0x25, OP_SIGMOID = 0x26, OP_DIV = 0x27,
    OP_SOFTPLUS = 0x28, OP_RSQRT = 0x29, OP_ADD_CONST = 0x2A, OP_MUL_CONST = 0x2B,
    OP_LUT_U8 = 0x30,
    OP_FOR_BEGIN = 0x40, OP_FOR_END = 0x41,
    OP_BARRIER_TG = 0x50,
    OP_TG_LOAD_F32 = 0x60, OP_TG_STORE_F32 = 0x61,
    OP_CMP_LT = 0x70, OP_IF_BEGIN = 0x72, OP_IF_END = 0x73,
    OP_U_TGID = 0x80, OP_U_LID = 0x81, OP_U_CONST = 0x82,
    OP_U_ADD = 0x83, OP_U_MUL = 0x84, OP_U_ADDK = 0x85,
    OP_U_MULK = 0x86, OP_U_ANDK = 0x87, OP_U_SHRK = 0x88, OP_U_SHLK = 0x89,
    OP_U_SHR = 0x8A, OP_U_LOAD_U8 = 0x8B, OP_U_LOAD_CONSTF = 0x8C,
    OP_U_LOAD_IN = 0x8D, OP_U_STORE_F32 = 0x8E, OP_U_TO_R = 0x8F,
    OP_R_TO_U = 0x90, OP_U_LOAD_F32 = 0x91, OP_U_ATOMIC_ADD = 0x92,
    OP_GRID_BARRIER = 0x93, OP_BASE_F32 = 0x94, OP_BASE_U8 = 0x95,
    OP_BASE_LUT = 0x96, OP_BASE_IN = 0x97, OP_U_LOAD_META = 0x98,
    OP_VTG_BEGIN = 0x99, OP_VTG_END = 0x9A, OP_LOAD_VACT = 0x9B,
    OP_GRID_EPILOGUE = 0x9C, OP_U_LOAD_U32 = 0x9D
};

typedef struct {
    int back_pc;
    int remain;
} VmLoop;

typedef struct {
    int pc;
    float r[16];
    uint32_t u[16];
    int sp;
    VmLoop loops[VM_MAX_LOOPS];
    size_t base_f32, base_u8, base_lut, base_in;
    int halted;
    int wait_tg, wait_grid;
} VmThread;

static inline uint32_t op_of(uint32_t w) { return w & 0xFFu; }
static inline uint32_t a_of(uint32_t w) { return (w >> 8) & 0xFu; }
static inline uint32_t b_of(uint32_t w) { return (w >> 12) & 0xFu; }
static inline uint32_t c_of(uint32_t w) { return (w >> 16) & 0xFu; }

static int extra_words(uint32_t op) {
    switch (op) {
    case OP_LOAD_F32: case OP_STORE_F32: case OP_LOAD_U8:
    case OP_LOAD_CONST: case OP_FMA: case OP_ADD_CONST: case OP_MUL_CONST:
    case OP_TG_LOAD_F32: case OP_TG_STORE_F32:
    case OP_U_CONST: case OP_U_ADDK: case OP_U_MULK: case OP_U_ANDK:
    case OP_U_SHRK: case OP_U_SHLK: case OP_U_LOAD_U8: case OP_U_LOAD_U32:
    case OP_U_LOAD_CONSTF: case OP_U_LOAD_IN: case OP_U_STORE_F32:
    case OP_U_LOAD_F32: case OP_U_ATOMIC_ADD: case OP_U_LOAD_META:
    case OP_GRID_BARRIER: case OP_GRID_EPILOGUE:
        return 1;
    case OP_VTG_BEGIN:
        return 2;
    default:
        return 0;
    }
}

static int next_pc(const uint32_t* code, int n, int pc) {
    if (pc < 0 || pc >= n) return n;
    uint32_t op = op_of(code[pc]);
    int npc = pc + 1 + extra_words(op);
    return npc > n ? n : npc;
}

/* Skip a nested FOR/IF block.  pc points just after the opening word. */
static int skip_block(const uint32_t* code, int n, int pc, uint32_t end_op) {
    int depth = 1;
    while (pc < n && depth > 0) {
        uint32_t op = op_of(code[pc]);
        if (op == OP_FOR_BEGIN || op == OP_IF_BEGIN) ++depth;
        else if (op == OP_FOR_END || op == OP_IF_END) --depth;
        pc = next_pc(code, n, pc);
    }
    (void)end_op;
    return pc;
}

typedef struct {
    float* mem;
    int mem_len;
    const float* in;
    int in_len;
    float* tg;
    int tg_floats;
    int n_threads;
    int tg_size;
    int n_groups;
    VmThread* threads;
} VmState;

static int check_idx(int64_t idx, int len, const char* what, char* err, int err_len) {
    if (idx < 0 || idx >= len) {
        if (err && err_len > 0)
            snprintf(err, (size_t)err_len, "CPU VM %s out of range: %lld", what, (long long)idx);
        return 0;
    }
    return 1;
}

/* Main interpreter. */
int nimf_cpu_vm_run(uint32_t* code, int n_words, float* mem, int mem_len,
                    float* in_f32, int in_len, int n_threads,
                    int tg_floats, int tg_size, char* err, int err_len) {
    if (err && err_len > 0) err[0] = '\0';
    if (!code || n_words <= 0 || !mem || mem_len <= 0 || n_threads <= 0 ||
        tg_size <= 0 || tg_floats < 0) {
        if (err && err_len > 0) snprintf(err, (size_t)err_len, "invalid CPU VM arguments");
        return -1;
    }
    if (n_threads > VM_MAX_THREADS) {
        if (err && err_len > 0) snprintf(err, (size_t)err_len, "CPU VM max threads %d", VM_MAX_THREADS);
        return -1;
    }
    int n_groups = (n_threads + tg_size - 1) / tg_size;
    size_t tg_total = (size_t)n_groups * (size_t)tg_floats;
    float* tg = NULL;
    if (tg_total > 0) {
        tg = (float*)calloc(tg_total, sizeof(float));
        if (!tg) { if (err && err_len > 0) snprintf(err, (size_t)err_len, "CPU VM tg alloc failed"); return -1; }
    }
    VmThread* threads = (VmThread*)calloc((size_t)n_threads, sizeof(VmThread));
    if (!threads) { free(tg); if (err && err_len > 0) snprintf(err, (size_t)err_len, "CPU VM thread alloc failed"); return -1; }
    for (int i = 0; i < n_threads; ++i) {
        threads[i].pc = 0;
        threads[i].r[0] = (float)i;
        threads[i].halted = 0;
    }

    const float* input = in_f32 ? in_f32 : mem;
    int input_len = in_f32 ? in_len : mem_len;
    int active = n_threads;
    uint32_t steps = 0;
    int result = 0;

    while (active > 0 && steps < VM_MAX_STEPS) {
        int progressed = 0;
        for (int tid = 0; tid < n_threads; ++tid) {
            VmThread* t = &threads[tid];
            if (t->halted || t->wait_tg || t->wait_grid) continue;
            int lid = tid % tg_size;
            int tgid = tid / tg_size;
            float* tgp = tg ? tg + (size_t)tgid * (size_t)tg_floats : NULL;
            int pc = t->pc;
            if (pc < 0 || pc >= n_words) {
                snprintf(err, (size_t)err_len, "CPU VM pc out of range (tid %d)", tid);
                result = -1; goto done;
            }
            uint32_t w = code[pc];
            uint32_t op = op_of(w);
            uint32_t a = a_of(w), b = b_of(w), c = c_of(w);
            int npc = pc + 1;
            progressed = 1;
            ++steps;

            switch (op) {
            case OP_NOP:
                break;
            case OP_HALT:
                t->halted = 1; --active;
                break;
            case OP_LOAD_F32: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t off = code[npc++];
                int64_t idx = (int64_t)(uint32_t)t->r[b] + (int64_t)off + (int64_t)t->base_f32;
                if (!check_idx(idx, mem_len, "load_f32", err, err_len)) { result = -1; goto done; }
                t->r[a] = mem[idx];
                break;
            }
            case OP_STORE_F32: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t off = code[npc++];
                int64_t idx = (int64_t)(uint32_t)t->r[b] + (int64_t)off + (int64_t)t->base_f32;
                if (!check_idx(idx, mem_len, "store_f32", err, err_len)) { result = -1; goto done; }
                mem[idx] = t->r[a];
                break;
            }
            case OP_LOAD_U8: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t off = code[npc++];
                int64_t idx = (int64_t)(uint32_t)t->r[b] + (int64_t)off + (int64_t)t->base_u8;
                if (!check_idx(idx, mem_len * 4, "load_u8", err, err_len)) { result = -1; goto done; }
                t->r[a] = (float)((const uint8_t*)mem)[idx];
                break;
            }
            case OP_LOAD_CONST: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t bits = code[npc++];
                float v; memcpy(&v, &bits, sizeof(v)); t->r[a] = v;
                break;
            }
            case OP_LOAD_F32_IDX: {
                int64_t idx = (int64_t)(uint32_t)t->r[b] + (int64_t)t->base_f32;
                if (!check_idx(idx, mem_len, "load_f32_idx", err, err_len)) { result = -1; goto done; }
                t->r[a] = mem[idx];
                break;
            }
            case OP_STORE_F32_IDX: {
                int64_t idx = (int64_t)(uint32_t)t->r[b] + (int64_t)t->base_f32;
                if (!check_idx(idx, mem_len, "store_f32_idx", err, err_len)) { result = -1; goto done; }
                mem[idx] = t->r[a];
                break;
            }
            case OP_LOAD_LID: t->r[a] = (float)lid; break;
            case OP_LOAD_TGID: t->r[a] = (float)tgid; break;
            case OP_ADD: t->r[a] = t->r[b] + t->r[c]; break;
            case OP_SUB: t->r[a] = t->r[b] - t->r[c]; break;
            case OP_MUL: t->r[a] = t->r[b] * t->r[c]; break;
            case OP_DIV: t->r[a] = t->r[b] / t->r[c]; break;
            case OP_FMA: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t rc = code[npc++] & 0xFu;
                t->r[a] = fmaf(t->r[b], t->r[c], t->r[rc]);
                break;
            }
            case OP_EXP: t->r[a] = expf(t->r[b]); break;
            case OP_NEG: t->r[a] = -t->r[b]; break;
            case OP_SIGMOID: t->r[a] = 1.0f / (1.0f + expf(-t->r[b])); break;
            case OP_SOFTPLUS: t->r[a] = logf(1.0f + expf(t->r[b])); break;
            case OP_RSQRT: t->r[a] = 1.0f / sqrtf(t->r[b]); break;
            case OP_ADD_CONST: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t bits = code[npc++]; float v; memcpy(&v, &bits, sizeof(v));
                t->r[a] = t->r[b] + v;
                break;
            }
            case OP_MUL_CONST: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t bits = code[npc++]; float v; memcpy(&v, &bits, sizeof(v));
                t->r[a] = t->r[b] * v;
                break;
            }
            case OP_LUT_U8: {
                int64_t idx = (int64_t)(uint32_t)t->r[b] + (int64_t)t->base_lut;
                if (!check_idx(idx, mem_len, "lut_u8", err, err_len)) { result = -1; goto done; }
                t->r[a] = mem[idx];
                break;
            }
            case OP_FOR_BEGIN: {
                int remain = (int)t->r[a];
                if (remain <= 0) {
                    npc = skip_block(code, n_words, npc, OP_FOR_END);
                } else {
                    if (t->sp >= VM_MAX_LOOPS) { snprintf(err, (size_t)err_len, "CPU VM loop stack overflow"); result = -1; goto done; }
                    t->loops[t->sp].back_pc = npc;
                    t->loops[t->sp].remain = remain - 1;
                    ++t->sp;
                }
                break;
            }
            case OP_FOR_END: {
                if (t->sp <= 0) { snprintf(err, (size_t)err_len, "CPU VM for/end underflow"); result = -1; goto done; }
                VmLoop* l = &t->loops[t->sp - 1];
                if (l->remain > 0) { --l->remain; npc = l->back_pc; }
                else --t->sp;
                break;
            }
            case OP_IF_BEGIN:
                if (t->r[a] == 0.0f) npc = skip_block(code, n_words, npc, OP_IF_END);
                break;
            case OP_IF_END:
                break;
            case OP_BARRIER_TG:
                t->wait_tg = 1;
                break;
            case OP_GRID_BARRIER:
                npc += 1; /* ctrl word */
                t->wait_grid = 1;
                break;
            case OP_GRID_EPILOGUE:
                npc += 1;
                break;
            case OP_TG_LOAD_F32: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t off = code[npc++];
                int64_t idx = (int64_t)(uint32_t)t->r[b] + (int64_t)off;
                if (tgp && idx >= 0 && idx < tg_floats) t->r[a] = tgp[idx];
                break;
            }
            case OP_TG_STORE_F32: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t off = code[npc++];
                int64_t idx = (int64_t)(uint32_t)t->r[b] + (int64_t)off;
                if (tgp && idx >= 0 && idx < tg_floats) tgp[idx] = t->r[a];
                break;
            }
            case OP_CMP_LT:
                t->r[a] = t->r[b] < t->r[c] ? 1.0f : 0.0f;
                break;
            case OP_U_TGID: t->u[a] = (uint32_t)tgid; break;
            case OP_U_LID: t->u[a] = (uint32_t)lid; break;
            case OP_U_CONST:
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                t->u[a] = code[npc++];
                break;
            case OP_U_ADD: t->u[a] = t->u[b] + t->u[c]; break;
            case OP_U_MUL: t->u[a] = t->u[b] * t->u[c]; break;
            case OP_U_SHR: t->u[a] = t->u[b] >> (t->u[c] & 31u); break;
            case OP_U_ADDK:
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                t->u[a] = t->u[b] + code[npc++];
                break;
            case OP_U_MULK:
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                t->u[a] = t->u[b] * code[npc++];
                break;
            case OP_U_ANDK:
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                t->u[a] = t->u[b] & code[npc++];
                break;
            case OP_U_SHRK:
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                t->u[a] = t->u[b] >> (code[npc++] & 31u);
                break;
            case OP_U_SHLK:
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                t->u[a] = t->u[b] << (code[npc++] & 31u);
                break;
            case OP_U_LOAD_U8: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t off = code[npc++];
                int64_t idx = (int64_t)t->u[b] + (int64_t)off + (int64_t)t->base_u8;
                if (!check_idx(idx, mem_len * 4, "u_load_u8", err, err_len)) { result = -1; goto done; }
                t->u[a] = (uint32_t)((const uint8_t*)mem)[idx];
                break;
            }
            case OP_U_LOAD_U32: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t off = code[npc++];
                int64_t idx = (int64_t)t->u[b] + (int64_t)off + (int64_t)t->base_u8;
                if (!check_idx(idx, mem_len * 4 - 3, "u_load_u32", err, err_len)) { result = -1; goto done; }
                uint32_t v; memcpy(&v, ((const uint8_t*)mem) + idx, sizeof(v)); t->u[a] = v;
                break;
            }
            case OP_U_LOAD_CONSTF: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t off = code[npc++];
                int64_t idx = (int64_t)t->u[b] + (int64_t)off + (int64_t)t->base_lut;
                if (!check_idx(idx, mem_len, "u_load_constf", err, err_len)) { result = -1; goto done; }
                t->r[a] = mem[idx];
                break;
            }
            case OP_U_LOAD_IN: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t off = code[npc++];
                int64_t idx = (int64_t)t->u[b] + (int64_t)off + (int64_t)t->base_in;
                if (!check_idx(idx, input_len, "u_load_in", err, err_len)) { result = -1; goto done; }
                t->r[a] = input[idx];
                break;
            }
            case OP_U_STORE_F32: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t off = code[npc++];
                int64_t idx = (int64_t)t->u[b] + (int64_t)off + (int64_t)t->base_f32;
                if (!check_idx(idx, mem_len, "u_store_f32", err, err_len)) { result = -1; goto done; }
                mem[idx] = t->r[a];
                break;
            }
            case OP_U_LOAD_F32: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t off = code[npc++];
                int64_t idx = (int64_t)t->u[b] + (int64_t)off + (int64_t)t->base_f32;
                if (!check_idx(idx, mem_len, "u_load_f32", err, err_len)) { result = -1; goto done; }
                t->r[a] = mem[idx];
                break;
            }
            case OP_U_ATOMIC_ADD: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t off = code[npc++];
                int64_t idx = (int64_t)t->u[b] + (int64_t)off + (int64_t)t->base_f32;
                if (!check_idx(idx, mem_len, "u_atomic_add", err, err_len)) { result = -1; goto done; }
                mem[idx] += t->r[a];
                break;
            }
            case OP_U_TO_R: t->r[a] = (float)t->u[b]; break;
            case OP_R_TO_U: t->u[a] = (uint32_t)t->r[b]; break;
            case OP_U_LOAD_META: {
                if (npc >= n_words) { result = -1; snprintf(err, (size_t)err_len, "CPU VM truncated instruction"); goto done; }
                uint32_t off = code[npc++];
                int64_t idx = (int64_t)t->u[b] + (int64_t)off + (int64_t)t->base_lut;
                if (!check_idx(idx, mem_len, "u_load_meta", err, err_len)) { result = -1; goto done; }
                uint32_t v; memcpy(&v, &mem[idx], sizeof(v)); t->u[a] = v;
                break;
            }
            case OP_BASE_F32: t->base_f32 = t->u[a]; break;
            case OP_BASE_U8: t->base_u8 = t->u[a]; break;
            case OP_BASE_LUT: t->base_lut = t->u[a]; break;
            case OP_BASE_IN: t->base_in = t->u[a]; break;
            case OP_VTG_BEGIN: npc += 2; break;
            case OP_VTG_END: break;
            case OP_LOAD_VACT: t->r[a] = 1.0f; break;
            default:
                snprintf(err, (size_t)err_len, "CPU VM unsupported opcode 0x%02X (tid %d pc %d)",
                         op, tid, pc);
                result = -1; goto done;
            }
            t->pc = npc;
        }

        /* Release threadgroup barriers whose whole group arrived. */
        for (int g = 0; g < n_groups; ++g) {
            int lo = g * tg_size;
            int hi = lo + tg_size; if (hi > n_threads) hi = n_threads;
            int all = 1, any = 0;
            for (int i = lo; i < hi; ++i) {
                if (!threads[i].halted && !threads[i].wait_tg) { all = 0; break; }
                if (threads[i].wait_tg) any = 1;
            }
            if (all && any) for (int i = lo; i < hi; ++i) threads[i].wait_tg = 0;
        }
        /* Release a grid barrier when every live thread arrived. */
        int grid_all = 1, grid_any = 0;
        for (int i = 0; i < n_threads; ++i) {
            if (!threads[i].halted && !threads[i].wait_grid) { grid_all = 0; break; }
            if (threads[i].wait_grid) grid_any = 1;
        }
        if (grid_all && grid_any) {
            for (int i = 0; i < n_threads; ++i) threads[i].wait_grid = 0;
        }
        if (!progressed) {
            snprintf(err, (size_t)err_len, "CPU VM deadlock (all threads waiting)");
            result = -1; goto done;
        }
    }
    if (active > 0 && steps >= VM_MAX_STEPS) {
        snprintf(err, (size_t)err_len, "CPU VM step limit exceeded");
        result = -1;
    }
    goto done;
done:
    free(threads);
    free(tg);
    return result;
}
