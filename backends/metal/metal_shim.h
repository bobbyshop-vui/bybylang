// metal_shim.h - C interface exposed to Nim for the Metal compute backend (macOS only).
#ifndef METAL_SHIM_H
#define METAL_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

// Trả về 1 nếu máy có GPU hỗ trợ Metal, 0 nếu không.
int metal_available(void);

// kernel_src: nguồn Metal Shading Language (đọc từ kernels/vecop_matmul.metal
// lúc Nim compile-time qua staticRead, KHÔNG hardcode trong file .m nữa).
// op: 0=add, 1=sub, 2=mul, 3=div. Trả về 1 nếu chạy thành công, 0 nếu lỗi.
// error_out: nếu không NULL, sẽ được gán trỏ tới chuỗi lỗi (cần free bằng free()).
int metal_vecop(const char* kernel_src, int op, const float* a, const float* b, float* c, int n, char** error_out);

// C(m x n) = A(m x k) * B(k x n), row-major. Trả về 1 nếu chạy thành công, 0 nếu lỗi.
int metal_matmul(const char* kernel_src, const float* a, const float* b, float* c, int m, int k, int n, char** error_out);

// Giống metal_matmul nhưng dùng MPSMatrixMultiplication (Metal Performance
// Shaders) của Apple thay vì kernel tile tự viết tay. MPS chọn thuật toán
// (simdgroup_matrix trên Apple Silicon, đường tối ưu riêng cho Intel/AMD)
// và autotune theo GPU thực tế đang chạy - luôn >= kernel tay viết, thường
// nhanh hơn nhiều. KHÔNG cần kernel_src (không dùng thư viện MSL tự biên
// dịch của mình cho đường này). Trả về 1 nếu chạy thành công, 0 nếu lỗi
// (vd. framework MetalPerformanceShaders không có / GPU không hỗ trợ) -
// caller PHẢI fallback về metal_matmul() khi hàm này trả 0.
int metal_matmul_mps(const float* a, const float* b, float* c, int m, int k, int n, char** error_out);

// Matmul truc tiep tren weight int4-asymmetric DA PACK (khong dequant ra
// fp32 tren CPU truoc). wq: packed bytes [N * ceil(K/2)]. scales/zeros:
// [N * nGroupsPerRow] (nGroupsPerRow = ceil(K/groupSize), groupSize<=0 nghia
// la 1 group/hang). Xem chu thich chi tiet trong vecop_matmul.metal
// (matmul_q4_kernel) truoc khi dung - CHUA CHAY THU TREN GPU THAT.
int metal_matmul_q4(const char* kernel_src, const float* a, const unsigned char* wq,
                     const float* scales, const float* zeros, float* c,
                     int m, int k, int n, int groupSize, int nGroupsPerRow, char** error_out);

// Chạy 2 phép matmul ĐỘC LẬP (a1*b1->c1 và a2*b2->c2) trong CÙNG một command
// buffer / một lần compile pipeline, chỉ commit+wait MỘT LẦN. Dùng khi cần
// dispatch 2 matmul không phụ thuộc lẫn nhau (vd. dW và dX trong backward)
// mà không muốn trả giá overhead của 2 lần commit+wait riêng lẻ.
// Trả về 1 nếu chạy thành công, 0 nếu lỗi.
int metal_matmul2(const char* kernel_src,
                   const float* a1, const float* b1, float* c1, int m1, int k1, int n1,
                   const float* a2, const float* b2, float* c2, int m2, int k2, int n2, char** error_out);

int metal_activation(const char* kernel_src, int op, const float* x, float* y, int n, float p, char** error_out);
int metal_softmax(const char* kernel_src, const float* x, float* y, int rows, int cols, char** error_out);
int metal_layernorm(const char* kernel_src, const float* x, const float* gamma, const float* beta, float* y, int rows, int cols, float eps, char** error_out);
int metal_embedding_lookup(const char* kernel_src, const float* table, const int* indices, float* y, int vocab, int dim, int num_indices, char** error_out);

// op: 0=relu, 1=sigmoid, 2=tanh, 3=apflu (dùng chung bảng mã với metal_activation).
// Gọi activation_backward_kernel trong vecop_matmul.metal (đã có sẵn kernel,
// trước đây CHƯA được bind ra hàm C -> gpubackend.nim không có đường gọi tới
// nên luôn rớt xuống CPU cho apflu backward dù chọn backend metal).
int metal_activation_backward(const char* kernel_src, int op, const float* x, const float* dy, float* dx, int n, char** error_out);

// Gọi layernorm_backward_kernel. dx: rows*cols phần tử, dgamma/dbeta: cols phần tử.
int metal_layernorm_backward(const char* kernel_src, const float* dy, const float* x,
                              const float* gamma, const float* beta,
                              float* dx, float* dgamma, float* dbeta,
                              int rows, int cols, float eps, char** error_out);

// Causal fused attention forward. q/k/v: B*H*S*D phần tử, o: B*H*S*D,
// s_matrix: B*H*S*S. `mask` không dùng trong kernel (causal áp qua vòng lặp
// tj<=ti giống bản CPU tham chiếu) nhưng vẫn nhận vào để chữ ký khớp
// beAttentionFused ở tầng backend.nim.
int metal_attention_fused(const char* kernel_src, const float* q, const float* k, const float* v,
                           float* o, float* s_matrix, int B, int H, int S, int D, float scale, char** error_out);

// Causal fused attention backward. dq/dk/dv: B*H*S*D phần tử MỖI cái, PHẢI
// được caller zero-init trước khi gọi (kernel dùng "+=" / atomic-add trên
// buffer output, không ghi đè hoàn toàn như forward).
int metal_attention_fused_backward(const char* kernel_src, const float* q, const float* k, const float* v,
                                    const float* s_matrix, const float* dy,
                                    float* dq, float* dk, float* dv,
                                    int B, int H, int S, int D, float scale, char** error_out);

// ─────────────────────────────────────────────────────────────────────────
// API resident: nhiều op mã hoá vào CÙNG 1 MTLCommandBuffer (giống metal_matmul2
// đã làm cho 2 matmul), chỉ commit+waitUntilCompleted MỘT LẦN ở
// metal_session_end(). Buffer trả về là opaque handle (con trỏ tới id<MTLBuffer>
// đã CFBridgingRetain) - giữ sống tới khi gọi metal_buffer_free.
// LƯU Ý: CHƯA CHẠY THỬ TRÊN GPU METAL THẬT (môi trường sinh code này là Linux,
// không có macOS/framework Metal để biên dịch+chạy) - viết bám sát 100% idiom
// đã có sẵn và ĐANG CHẠY ĐƯỢC trong metal_vecop/metal_matmul/metal_matmul2 ở
// trên (cùng cách tạo buffer, cùng cách lấy pipeline, cùng cách dispatch), chỉ
// khác ở chỗ dùng chung 1 command buffer thay vì tạo mới mỗi lệnh. Bắt buộc
// phải build + test trên máy Mac thật trước khi dùng cho train thật.
typedef void* MetalBufferHandle;

// Bắt đầu 1 phiên: tạo command buffer + encoder mới, lưu vào biến static.
// Gọi trước chuỗi op resident đầu tiên của 1 layer/1 forward pass.
int metal_session_begin(const char* kernel_src, char** error_out);

// Upload dữ liệu host -> MTLBuffer (memcpy CPU, Apple Silicon là unified memory
// nên KHÔNG có DMA H2D thật như CUDA rời - vẫn tách hàm riêng cho rõ ràng
// và để chạy đúng trên Mac Intel có GPU rời nếu có, vì lúc đó storage mode
// khác đi driver mới thật sự cần transfer).
MetalBufferHandle metal_upload(const float* data, int n);
// Session-only borrowed upload: reuses page-aligned host storage when Metal
// accepts it. The caller must keep `data` alive and unchanged until the
// session has ended and the buffer has been freed. Falls back to a copied
// Shared buffer when zero-copy is not supported for this address/length.
MetalBufferHandle metal_upload_borrowed(const float* data, int n);
MetalBufferHandle metal_upload_indices(const int* data, int n);
// Cấp buffer output resident (kernel sẽ ghi đè), CHƯA init dữ liệu.
MetalBufferHandle metal_alloc_scratch(int n);

// THÊM MỚI: MTLHeap - dùng khi cần cấp phát NHIỀU buffer cùng lúc, cùng
// vòng đời (điển hình: KV-cache resident, mỗi layer cần nKVHeads*2 buffer
// K/V). metal_alloc_scratch() gọi [gDevice newBufferWithLength:] TRỰC TIẾP
// mỗi lần - đây là 1 lượt cấp phát device-level đầy đủ (map trang nhớ
// shared CPU/GPU), có phí cố định đáng kể trên driver Intel Iris cũ. Với
// KV resident: nKVHeads*2*nLayers lần gọi riêng lẻ (vd 32 head*2*32 layer =
// 2048 lần) đo thực tế chiếm hàng chục giây NGAY LÚC KHỞI ĐỘNG (trước token
// đầu tiên) - đây là bug hiệu năng THẬT, không phải cảm giác.
// Fix: cấp 1 vùng lớn DUY NHẤT bằng MTLHeap (1 lần/layer, hoặc 1 lần cho cả
// model), rồi "suballocate" từng buffer con từ heap đó bằng
// metal_alloc_from_heap() - suballocate từ heap đã reserve sẵn RẺ HƠN NHIỀU
// so với gọi thẳng device mỗi lần (đúng khuyến nghị chính thức của Apple
// trong Metal Best Practices Guide, mục "Heaps", cho đúng use-case "nhiều
// buffer cùng cỡ, cùng vòng đời"). Vẫn trả về MetalBufferHandle bình
// thường ở đầu ra - MỌI call site đang dùng metal_alloc_scratch() ở nơi
// khác (matmul/attention/...) KHÔNG cần đổi gì, vì buffer suballocate từ
// heap vẫn là 1 id<MTLBuffer> hợp lệ dùng được y hệt buffer cấp trực tiếp.
void* metal_alloc_heap(long bytes);
MetalBufferHandle metal_alloc_from_heap(void* heap, int n);
void metal_free_heap(void* heap);
// Encode thêm 1 op vào command buffer đang mở (KHÔNG commit) - trả về 1/0.
int metal_vecop_enc(int op, MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int n);
int metal_activation_enc(int op, MetalBufferHandle x, MetalBufferHandle y, int n);
int metal_scale_enc(MetalBufferHandle x, float scalar, int n);
int metal_softmax_enc(MetalBufferHandle x, MetalBufferHandle y, int rows, int cols);
int metal_copy_into_enc(MetalBufferHandle src, MetalBufferHandle dst, int srcOffsetElems, int dstOffsetElems, int n);
int metal_rmsnorm_enc(MetalBufferHandle x, MetalBufferHandle gamma,
                       MetalBufferHandle y, int rows, int cols, float eps);
int metal_layernorm_enc(MetalBufferHandle x, MetalBufferHandle gamma, MetalBufferHandle beta,
                         MetalBufferHandle y, int rows, int cols, float eps);
int metal_embedding_lookup_enc(MetalBufferHandle table, MetalBufferHandle indices,
                                MetalBufferHandle y, int vocab, int dim, int num_indices);
int metal_matmul_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int m, int k, int n);
// C(m,n) = A(m,k) @ B(n,k)^T - B truyen vao GIU NGUYEN layout goc [n,k]
// (khong transpose CPU truoc). Dung cho attention decode (K/V-cache).
int metal_matmul_nt_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c, int m, int k, int n);
// Resident prefix GEMM. C may be a column tile of a wider row-major matrix.
int metal_matmul_mps_nt_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle c,
                            int m, int k, int n, int cRowStride, int cColumnOffset);
int metal_mps_supported(void);
// Runtime machine check. Returns 1 on Apple Silicon (including Rosetta), 0 otherwise.
int metal_is_apple_silicon(void);
// Tune Q4 workgroup/kernel geometry outside an open inference session.
int metal_prepare_q4_tuned(MetalBufferHandle w, MetalBufferHandle scales,
    MetalBufferHandle zeros, int k, int n, int groupSize, int groupsPerRow, int m);
// RoPE rotate_half in-place trong session đang mở. x: [T,Hn,D]. cosT/sinT: [T,D/2]
// (đã cắt đúng đoạn cần dùng phía CPU trước khi upload). Không đọc CPU giữa chừng.
int metal_rope_enc(MetalBufferHandle x, MetalBufferHandle cosT, MetalBufferHandle sinT,
                    int T, int Hn, int D);
// Encode attention_fused_kernel vào session đang mở (KHÔNG commit). q,k,v đã
// là buffer resident (metal_upload trước đó), o/s_matrix là buffer scratch
// (metal_alloc_scratch). S phải <= 256 (giới hạn kernel, giống bản non-session).
int metal_attention_fused_enc(MetalBufferHandle q, MetalBufferHandle k, MetalBufferHandle v,
                               MetalBufferHandle o, MetalBufferHandle s_matrix,
                               int B, int H, int S, int D, float scale);
// MOI THEM: ban flash (online-softmax, khong s_matrix) cua attention_fused
// - dung cho INFERENCE prefill. D (head_dim) phai <= 128. KHONG dung cho
// training (backward can s_matrix -> van goi metal_attention_fused_enc cu).
int metal_attention_fused_flash_enc(MetalBufferHandle q, MetalBufferHandle k, MetalBufferHandle v,
                                     MetalBufferHandle o,
                                     int B, int H, int S, int D, float scale);
// MỚI THÊM: attention decode cho GQA - 1 query/head (token mới), K/V dùng
// chung S dòng cho cả H head trong 1 nhóm GQA. Không giới hạn S<=256 (scores
// là buffer scratch H*S cấp phát đúng kích thước thực tế).
int metal_attention_decode_gqa_enc(MetalBufferHandle q, MetalBufferHandle k, MetalBufferHandle v,
                                    MetalBufferHandle o, MetalBufferHandle scores,
                                    int H, int S, int D, float scale);
// THÊM MỚI: attention decode FUSED cho TOÀN BỘ H head trong ĐÚNG 1 lệnh
// dispatch (thay vì vòng lặp CPU 32 head x 5 lệnh dispatch/head như trước).
// K/V phải ở layout LIÊN TỤC theo head: buffer kích thước HKV*maxSeqLen*D,
// offset head hkv = hkv*maxSeqLen*D (dùng kResidentCombined/vResidentCombined
// bên Nim, KHÔNG dùng kResident/vResident cũ vì suballocate từ heap không
// đảm bảo khoảng cách đều giữa các head). Q đã nhân RoPE+scale sẵn trước khi
// gọi hàm này (giống hệt đường cũ). Kernel dùng thuật toán softmax cổ điển
// 4 pha, viết mới hoàn toàn - KHÔNG dùng chung code với
// metal_attention_decode_gqa_enc (kernel đó đã bị ghi nhận sinh sai).
int metal_attention_decode_all_heads_enc(MetalBufferHandle q, MetalBufferHandle kAll, MetalBufferHandle vAll,
                                          MetalBufferHandle o,
                                          int H, int HKV, int D, int S, int maxSeqLen);
// Upload byte thô (weight int4 đã pack) -> MTLBuffer resident, dùng cho
// metal_matmul_q4_enc bên dưới. Handle sống tới khi gọi metal_buffer_free -
// gọi UPLOAD 1 LẦN lúc load model, giữ handle suốt vòng đời process, KHÔNG
// gọi lại mỗi token (đó là toàn bộ mục đích của resident weight).
// SUA (bug thật gặp lúc chạy): metal_upload/metal_upload_bytes (dùng cho
// resident weight - gọi TRƯỚC bất kỳ metal_session_begin/metal_matmul nào,
// ngay lúc load model) trước đây KHÔNG gọi ensureMetalInit() như mọi hàm
// khác trong file này -> gDevice vẫn NULL -> newBufferWithBytes trên device
// nil trả về nil -> "tao buffer FAILED" hàng loạt (thấy trực tiếp trên log
// máy Intel Iris thật). Thêm hàm init tường minh, gọi 1 LẦN trước khi upload
// resident đầu tiên - ensureMetalInit() dùng dispatch_once nên gọi nhiều lần
// vẫn an toàn/rẻ (chỉ init thật ở lần gọi đầu).
int metal_init(const char* kernel_src);

MetalBufferHandle metal_upload_bytes(const unsigned char* data, int n);
// THÊM MỚI: bản Private-storage - CHỈ dùng cho resident weight (GPU-only
// sau lúc upload). Dùng metal_upload/metal_upload_bytes (Shared) như cũ cho
// mọi buffer activation (CPU còn đụng: upload mỗi token hoặc đọc kết quả về).
MetalBufferHandle metal_upload_private(const float* data, int n);
MetalBufferHandle metal_upload_bytes_private(const unsigned char* data, int n);
// Bản resident/session của metal_matmul_q4 (xem metal_matmul_q4 phía trên):
// wq/scales/zeros là handle đã upload SẴN 1 LẦN (metal_upload_bytes/metal_upload),
// KHÔNG memcpy lại mỗi lần gọi hàm này - chỉ activation `a` mới nhỏ/đổi mỗi
// token. Encode vào session đang mở (giữa metal_session_begin/metal_session_end),
// giống hệt metal_matmul_enc nhưng đọc thẳng int4 packed thay vì fp32.
int metal_matmul_q4_enc(MetalBufferHandle a, MetalBufferHandle wq, MetalBufferHandle scales,
                         MetalBufferHandle zeros, MetalBufferHandle c,
                         int m, int k, int n, int groupSize, int nGroupsPerRow);
// THÊM MỚI: matmul Q4 CÓ BATCH M THẬT, chỉ dùng cho M>1 (prefill). Kernel
// đọc mỗi byte weight ĐÚNG 1 LẦN, dùng lại cho MBLOCK=4 hàng liên tiếp -
// khác hẳn metal_matmul_q4_enc (dispatch Y=M, mỗi hàng đọc lại toàn bộ
// weight độc lập - đúng như thiết kế cho M=1/decode, nhưng cực kém hiệu
// quả khi bị gọi với M>1). Xem matmul_q4_batched.metal.
int metal_matmul_q4_batched_enc(MetalBufferHandle a, MetalBufferHandle wq, MetalBufferHandle scales,
                                 MetalBufferHandle zeros, MetalBufferHandle c,
                                 int m, int k, int n, int groupSize, int nGroupsPerRow);

// FP4/FP8/quant 1-byte/weight: wq=uchar[N*K], lut=float[256] (buildDecodeLUT*scale).
// Kernel: matmul_fp4_lut_kernel_ndst2_paired (vecop_matmul.metal). Session+TG nhu Q4.
int metal_matmul_fp4_lut_enc(MetalBufferHandle a, MetalBufferHandle wq, MetalBufferHandle lut,
                              MetalBufferHandle c, int m, int k, int n);

// MỚI THÊM: bản activation_enc có tham số p (dùng cho "pow", opcode 7).
// metal_activation_enc (chữ ký cũ, không p) giữ nguyên, gọi thẳng hàm này
// với p=0.0 để không phá call site relu/sigmoid/tanh/apflu hiện tại.
int metal_activation_enc_p(int op, MetalBufferHandle x, MetalBufferHandle y, int n, float p);
// activation_backward_kernel trong session - dùng cho apflu_backward
// (op=3, alpha/beta cố định 0.1/1.0 giống metalApfluBackward immediate).
int metal_activation_backward_enc(int op, MetalBufferHandle x, MetalBufferHandle dy, MetalBufferHandle dx, int n);
int metal_layernorm_backward_enc(MetalBufferHandle dy, MetalBufferHandle x,
                                  MetalBufferHandle gamma, MetalBufferHandle beta,
                                  MetalBufferHandle dx, MetalBufferHandle dgamma, MetalBufferHandle dbeta,
                                  int rows, int cols, float eps);
int metal_attention_fused_backward_enc(MetalBufferHandle q, MetalBufferHandle k, MetalBufferHandle v,
                                        MetalBufferHandle s_matrix, MetalBufferHandle dy,
                                        MetalBufferHandle dq, MetalBufferHandle dk, MetalBufferHandle dv,
                                        int B, int H, int S, int D, float scale);
int metal_gather_enc(MetalBufferHandle src, MetalBufferHandle idx, MetalBufferHandle dst, int n);
int metal_argmax_enc(MetalBufferHandle x, MetalBufferHandle outVal, MetalBufferHandle outIdx,
                      int outer, int dimSize, int inner);
int metal_compare_enc(MetalBufferHandle a, MetalBufferHandle b, MetalBufferHandle mask, int op, int n);
int metal_dropout_mask_enc(MetalBufferHandle mask, unsigned int seed, float p, float scale, int n);
int metal_permute_enc(MetalBufferHandle src, MetalBufferHandle dst,
                       MetalBufferHandle newStrides, MetalBufferHandle oldStrides,
                       MetalBufferHandle dimsMap, int ndim, int n);
int metal_split_slice_enc(MetalBufferHandle src, MetalBufferHandle dst,
                           int dimTotal, int thisSize, int offset, int inner, int n);
int metal_concat_gather_enc(MetalBufferHandle flatSrc, MetalBufferHandle srcBase,
                             MetalBufferHandle srcDimSize, MetalBufferHandle srcDimOffset,
                             int numTensors, MetalBufferHandle dst, int totalDim, int inner, int n);
// Commit + wait MỘT LẦN cho toàn bộ op đã encode từ lúc metal_session_begin.
// Sau khi gọi hàm này, mọi MetalBufferHandle trong phiên đã có dữ liệu đúng,
// đọc được bằng metal_buffer_read.
int metal_session_end(void);
// MOI: "commitAndContinue" - commit buffer HIEN TAI (KHONG cho -
// khong waitUntilCompleted) roi mo NGAY buffer moi de tiep tuc encode.
// Xem giai thich day du (hoc theo MPSCommandBuffer.commitAndContinue cua
// Apple / SyncType::COMMIT_ADAPTIVE cua PyTorch MPS) trong metal_shim.m.
// Goi dinh ky giua 1 session dai (vi du sau moi N layer) de gioi han kich
// thuoc TUNG buffer rieng le (tranh watchdog) MA KHONG tra gia dong bo
// CPU<->GPU moi lan (khac voi goi metal_session_end() + metal_session_begin()
// lai giua chung - cach do CHAM vi la 1 lan cho THAT). Tra ve 1 neu thanh
// cong, 0 neu that bai (vi du chua co session dang mo).
int metal_session_commit_continue(void);
// THÊM MỚI: lấy mô tả NSError THẬT của lần metal_session_end() gần nhất bị
// fail (commit/waitUntilCompleted không Completed) - trước đây lỗi này CHỈ
// in ra NSLog (Console/syslog), không truyền ngược được về Nim, khiến caller
// (nim_inference.nim, đường Mode 2/Mode 3) chỉ biết "sessionEnd that bai"
// chung chung, không biết lý do thật (vd. timeout watchdog GPU, out of
// memory, v.v.). Trả về chuỗi rỗng "" nếu lần gần nhất KHÔNG fail (hoặc
// chưa từng gọi metal_session_end nào). Con trỏ trả về sống tới lần
// metal_session_end() TIẾP THEO (static NSString giữ strong ref) - Nim
// phải copy ra string ngay (dùng $cstring), không giữ con trỏ lâu dài.
const char* metal_get_last_session_error(void);
// MOI: tao lai MTLCommandQueue (gQueue) khi bi "poisoned" sau 1 loi GPU that
// (Timeout/StackOverflow/PageFault...). Xem giai thich chi tiet trong
// metal_shim.m::metal_recover_queue. Tra ve 1 neu thanh cong, 0 neu that bai.
// Goi ngay sau khi metal_session_end() tra ve 0 (that bai) TRUOC KHI mo
// session moi/doc ket qua - neu khong, moi session sau se tiep tuc bi
// "Ignored" vinh vien va sessionRead se tra ve du lieu CU/RAC.
int metal_recover_queue(void);
// Đọc dữ liệu về host (chỉ hợp lệ SAU metal_session_end của phiên tạo ra handle đó).
int metal_buffer_read(MetalBufferHandle h, float* outData, int n);
// Giải phóng ngay khi không còn cần (vd. activation trung gian sau khi dùng
// xong trong backward) - an toàn gọi bất cứ lúc nào, kể cả giữa 1 phiên đang
// mở, vì buffer chỉ thật sự release() sau khi ARC thấy không còn tham chiếu
// nào trong command buffer đang chạy (Metal tự retain buffer khi encode).
void metal_buffer_free(MetalBufferHandle h);

// GPU HOÁ THẬT (vòng 2): gather (permute/concat/split/stack), argmax
// (giá trị+chỉ số), compare (bitmask), dropout RNG - kernel đã nối vào
// cuối vecop_matmul.metal ("gather_kernel"/"argmax_kernel"/"compare_kernel"
// /"dropout_rng_kernel"), dùng chung kMetalKernelSrc, không cần library
// riêng. srcLen là số phần tử THẬT của buffer nguồn (idx[] chỉ tới trong
// đó) - cần tách riêng khỏi n (số phần tử OUTPUT/idx) vì với concat, buffer
// nguồn gộp nhiều tensor nên dài hơn từng lần gather riêng lẻ.
int metal_gather(const char* kernelSrc, const float* src, int srcLen,
                  const int* idx, float* dst, int n, char** error_out);
int metal_argmax(const char* kernelSrc, const float* x, float* outVal,
                  int* outIdx, int outer, int dimSize, int inner, char** error_out);
int metal_compare(const char* kernelSrc, const float* a, const float* b,
                   unsigned char* mask, int op, int n, char** error_out);
int metal_dropout_mask(const char* kernelSrc, float* mask,
                        unsigned int seed, float p, float scale, int n, char** error_out);

// GPU HOÁ THẬT (vòng 3): permute/split/concat - device tự giải mã chỉ số
// từ bảng nhỏ O(rank)/O(numTensors), không có mảng chỉ số O(n) build trên
// host (xem permute_strided/split_slice/concat_gather trong
// vecop_matmul.metal).
int metal_permute(const char* kernelSrc, const float* src, float* dst,
                   const int* newStrides, const int* oldStrides,
                   const int* dimsMap, int ndim, int n, char** error_out);
int metal_split_slice(const char* kernelSrc, const float* src, int srcLen, float* dst,
                       int dimTotal, int thisSize, int offset, int inner, int n, char** error_out);
int metal_concat_gather(const char* kernelSrc, const float* flatSrc, int flatLen,
                         const int* srcBase, const int* srcDimSize,
                         const int* srcDimOffset, int numTensors,
                         float* dst, int totalDim, int inner, int n, char** error_out);

/* ============================================================
 * Session DƯỚI CHÂN — map 1:1 MTLComputeCommandEncoder
 * Dán các dòng này vào metal_shim.h TRƯỚC #ifdef __cplusplus / #endif cuối.
 * ============================================================ */

/* Chọn kernel đã compile trong gLibrary (setComputePipelineState). */
int metal_enc_pipeline(const char* kernelName);

/* Bind MTLBuffer vào [[buffer(index)]]. offsetBytes thường = 0. */
int metal_enc_bind_buffer(int index, MetalBufferHandle buf, int offsetBytes);

/* Constant nhỏ qua setBytes (int / float / raw). */
int metal_enc_bind_int(int index, int v);
int metal_enc_bind_float(int index, float v);
int metal_enc_bind_bytes(int index, const void* data, int nBytes);

/* dispatchThreadgroups — tgX=0 => dùng threadExecutionWidth của pipeline hiện tại. */
int metal_enc_dispatch(int gridX, int gridY, int gridZ, int tgX, int tgY, int tgZ);

/* dispatchThreads (khi kernel dùng thread_position_in_grid thay vì threadgroup). */
int metal_enc_dispatch_threads(int threadsX, int threadsY, int threadsZ,
                               int tgX, int tgY, int tgZ);

/* memoryBarrierWithScope: MTLBarrierScopeBuffers */
int metal_enc_barrier(void);

/* setThreadgroupMemoryLength:atIndex: — PHẢI gọi trước mỗi dispatch cho
 * kernel JIT (lowerBytecodeToMetal phía Nim luôn khai báo tham số
 * threadgroup(0) không điều kiện, kể cả kernel không dùng threadgroup
 * memory thật — thiếu bước set length này Metal sẽ ném assertion lúc
 * dispatchThreads). index=0, nBytes=4 là đủ cho kernel không dùng TG
 * memory thật; kernel có dùng reduction (RMSNorm...) cần nBytes lớn hơn. */
int metal_enc_set_threadgroup_bytes(int index, int nBytes);

/* threadExecutionWidth của pipeline đang chọn (sau metal_enc_pipeline). */
int metal_enc_simd_width(void);

/* maxTotalThreadsPerThreadgroup của pipeline hiện tại. */
int metal_enc_max_tg_threads(void);



/* JIT: compile Metal source luc runtime (bytecode lower -> kernel moi). */
int metal_jit_compile(const char* source, const char* kernelName);


int metal_device_max_tg_threads(void);
int metal_device_max_tg_bytes(void);

#ifdef __cplusplus
}
#endif

#endif
