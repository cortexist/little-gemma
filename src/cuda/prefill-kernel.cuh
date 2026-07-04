// prefill-kernel.cuh — the prompt phase, split from model-cuda.cuh (same
// single-include unit, included at the point the model_prefill* entries used
// to sit; declaration order and codegen unchanged). The pure-prefill
// machinery lives here: the K-sharing wide-chunk attention, the tensor-core
// flash kernel, chunk_layers and the forward_chunk* forms, and the
// model_prefill* entry points. The decode/chunk twin kernels (rmsnorm/_n,
// attn/_n, kv_write/_n, geglu/_n) stay paired in model-cuda.cuh — each twin
// documents the other, and the split-for-measured-reasons story reads best
// side by side.
#ifndef PREFILL_KERNEL_CUH
#define PREFILL_KERNEL_CUH

// K/V-SHARING prefill attention. The per-query kernels above launch one block
// per (head, query); each re-reads the whole K/V prefix, so a chunk's B queries
// read it B times — and at a long context K/V exceeds L2, so that redundancy is
// the dominant prefill-attention cost (the profile put it at 28% E4B / 43% 12B).
// Here a block owns a tile of QT queries for one head: each K/V row is staged
// into shared ONCE per block and reused by all QT warps (one warp per query,
// online-softmax state in registers, exactly like d_attn). Global K/V reads
// drop ~QT×; the math per query is unchanged (validated in .scratch/
// attn_kvshare_test.cu vs a double reference). PREFILL ONLY — gated on B>2 so
// the B=2 MTP verify keeps the per-query path and its decode-matching argmax.
// The float combine order differs from the per-query kernel (one warp now sums
// all timesteps of its query, vs warps splitting them), so this is the same
// relaxed class as the online-softmax step; the quantize epilogue calls the
// identical d_quant_group, so only attention's own reassociation differs.
template <int QT, bool RING, typename KT>
__global__ static void attn_kvshare_n_kernel(float *xb, const float *q, const KT *Kc, const KT *Vc,
                                             int hd, int kv_dim, int gqa, const int *d_pos, int window, int seq,
                                             int B, int n_head, struct actq aq) {
    const int hh = blockIdx.x, kvh = hh / gqa;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int qbase = blockIdx.y * QT, b = qbase + warp;  // this warp's query (chunk row)
    const int pos = *d_pos + b;                           // its absolute position (valid if b<B)
    const int start = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0;
    const int nv = hd / 32;
    extern __shared__ float katt[];                       // sK[hd], sV[hd], sO[QT*hd]
    float *sK = katt, *sV = katt + hd, *sO = katt + 2 * hd;
    const float *qh = q + (size_t)b * n_head * hd + (size_t)hh * hd;   // deref only when b<B
    float acc[ATTN_HD_MAX / 32];
    #pragma unroll
    for (int j = 0; j < ATTN_HD_MAX / 32; j++) acc[j] = 0.0f;
    float m = -1e30f, s = 0.0f;

    // positions this block streams: from the tile's earliest window-start to its
    // latest query position. (For full layers window==0 -> from 0.)
    int lastb = qbase + QT - 1; if (lastb > B - 1) lastb = B - 1;
    int rhi = *d_pos + lastb;
    int rlo = 0;
    if (RING && window > 0) { rlo = *d_pos + qbase - window + 1; if (rlo < 0) rlo = 0; }
    for (int r = rlo; r <= rhi; r++) {
        int rr = RING ? r % seq : r;                      // ring row of absolute position r
        if (sizeof(KT) == 2) {                            // f16: one 32-bit load = 2 elements
            const __half2 *K2 = (const __half2 *)(const void *)(Kc + (size_t)rr * kv_dim + (size_t)kvh * hd);
            const __half2 *V2 = (const __half2 *)(const void *)(Vc + (size_t)rr * kv_dim + (size_t)kvh * hd);
            for (int i = threadIdx.x; i < hd / 2; i += blockDim.x) {
                float2 kf = __half22float2(K2[i]); sK[2 * i] = kf.x; sK[2 * i + 1] = kf.y;
                float2 vf = __half22float2(V2[i]); sV[2 * i] = vf.x; sV[2 * i + 1] = vf.y;
            }
        } else {
            const KT *kt = Kc + (size_t)rr * kv_dim + (size_t)kvh * hd;
            const KT *vt = Vc + (size_t)rr * kv_dim + (size_t)kvh * hd;
            for (int i = threadIdx.x; i < hd; i += blockDim.x) { sK[i] = (float)kt[i]; sV[i] = (float)vt[i]; }
        }
        __syncthreads();
        if (b < B && r >= start && r <= pos) {            // warp-uniform: causal + window mask
            float sc = 0.0f;
            for (int i = lane; i < hd; i += 32) sc += qh[i] * sK[i];
            for (int o = 16; o > 0; o >>= 1) sc += __shfl_down_sync(0xffffffffu, sc, o);
            sc = __shfl_sync(0xffffffffu, sc, 0);
            float mn = fmaxf(m, sc), corr = expf(m - mn), e = expf(sc - mn);
            s = s * corr + e;
            #pragma unroll
            for (int j = 0; j < ATTN_HD_MAX / 32; j++) if (j < nv) acc[j] = acc[j] * corr + e * sV[lane + j * 32];
            m = mn;
        }
        __syncthreads();                                  // before next r overwrites sK/sV
    }

    if (b < B) {
        float inv = 1.0f / s;
        float *outh = xb + (size_t)b * n_head * hd + (size_t)hh * hd;
        #pragma unroll
        for (int j = 0; j < ATTN_HD_MAX / 32; j++)
            if (j < nv) { float v = acc[j] * inv; outh[lane + j * 32] = v; sO[warp * hd + lane + j * 32] = v; }
        __syncwarp();                                     // sO visible across the warp
        if (aq.xq) {
            int gbase = (int)(((size_t)b * n_head * hd + (size_t)hh * hd) / 32);
            for (int g = lane; g < nv; g += 32) d_quant_group(sO + warp * hd + g * 32, gbase + g, aq);
        }
    }
}

// ==== tensor-core FLASH ATTENTION (prefill, B>2) ============================
// The attn_*_n kernels above are online-softmax with SCALAR dots — on the Orin
// prefill profile, attention is ~40% of TTFT and those dots leave the tensor
// cores idle. This is our specialized TC flash attention: per (query,head),
// out = softmax(Q.K^T).V (scale 1.0, no softcap, causal/windowed), Q/K/V already
// normed+roped. One CTA per head, 8 warps. KEY Orin choice: K/V are NOT staged
// to shared (read from the L2-cached cache) so shared stays small (~38KB at
// hd=512) and the 8-SM Orin seats ~4 CTAs — avoiding the 1-CTA/SM occupancy trap
// that sank the tiled MMQ. 8-warp QK^T computes one S[16q x 8k] tile FULL-hd (no
// cross-warp reduce) -> shared; online softmax (warp per 4 query rows); PV is
// hd-SPLIT (warp w owns hd[w*HDW,+HDW)) with V transposed via scalar reads.
// m16n8k16.f16.f16.f32: Q,P and the cache rows feed the mma as f16 (f32 SWA KV
// is converted) — precision is the f16-flash/quality-equivalent class (same as
// the shipped f16-KV step, ~2e-3 vs an f64 ref, validated in .scratch/flash_test
// .cu), NOT bit-identical. Writes f32 xb; the dispatch quantizes via act_quantize
// (the C-tile scatter doesn't give the contiguous 32-groups the epilogue needs).
// PREFILL ONLY, B>2 — the B<=2 MTP verify keeps the per-query path (bit-exact
// argmax). LG_NO_FLASH falls back to the attn_*_n kernels.
template<typename KT> __device__ __forceinline__ uint32_t fa_ld2(const KT *p);
template<> __device__ __forceinline__ uint32_t fa_ld2<__half>(const __half *p){ return *(const uint32_t *)p; }
template<> __device__ __forceinline__ uint32_t fa_ld2<float>(const float *p){
    return (uint32_t)__half_as_ushort(__float2half(p[0])) | ((uint32_t)__half_as_ushort(__float2half(p[1]))<<16); }
template<typename KT> __device__ __forceinline__ __half fa_rd1(const KT *p);
template<> __device__ __forceinline__ __half fa_rd1<__half>(const __half *p){ return *p; }
template<> __device__ __forceinline__ __half fa_rd1<float>(const float *p){ return __float2half(*p); }
__device__ __forceinline__ uint32_t fa_pk(__half a,__half b){ return (uint32_t)__half_as_ushort(a) | ((uint32_t)__half_as_ushort(b)<<16); }
__device__ __forceinline__ void fa_mma(float &c0,float &c1,float &c2,float &c3,
        uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        :"+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1)); }

// seg/bidir_hi carry Gemma's modality mask: seg[abs_pos] is a media-span id (0 for
// text), and within a span attention is BIDIRECTIONAL (a patch sees its whole frame,
// not just earlier patches). seg==NULL -> the kernel is byte-identical causal (text,
// decode, verify). For a media chunk the key loop extends to bidir_hi (the span's
// last position, which can sit past a query's own position) and same-span keys are
// unmasked regardless of causal/window — that is the frame attending to itself.
// G packs the G query-heads of one KV group into a single CTA (G=2 on gemma's
// gqa-2 SWA layers): a 32*G-row Q tile over the same 32 query positions, so
// each staged K block and each strided V fragment is fetched ONCE for all G
// heads — the kernel is mio-instruction-count bound (Orin SM 25.8%, mio 4.7;
// A5000 mio 5.2), and G=2 halves that count. Rows [0,32) are head head0's 32
// queries, rows [32,64) head0+1's; a row's mask depends only on its query
// POSITION, so row b and b^32 mask identically. G=1 is byte-for-byte the old
// kernel (every loop collapses to the old bounds). Shared memory is dynamic
// (G=2 at hd 256 needs ~62 KB, past the 48 KB static limit).
template<int HD, bool RING, typename KT, int G>
__global__ static void __launch_bounds__(256, 2)
flash_attn_n_kernel(float *xb, const float *q, const KT *Kc, const KT *Vc,
                    int kv_dim, int gqa, const int *d_pos, int window, int seq, int B, int n_head,
                    const int *seg, int bidir_hi){
    const int FKB = 32, FHDW = HD/8, ROWS = 32*G;          // keys/block; PV hd-slice per warp
    const bool KSTAGE = HD == 256 && sizeof(KT) == 4;      // see the K-stage note below
    extern __shared__ unsigned char fsh[];
    __half *sQ = (__half*)fsh;                                            // [ROWS*HD]
    float  *sS = (float*)(fsh + (size_t)ROWS*HD*2);                       // [ROWS*FKB]
    __half *sP = (__half*)(fsh + (size_t)ROWS*HD*2 + ROWS*FKB*4);         // [ROWS*FKB]
    float  *sm = (float*)(fsh + (size_t)ROWS*HD*2 + ROWS*FKB*6);          // [ROWS] x3
    float  *sl = sm + ROWS, *sc = sl + ROWS;
    __half *sK = (__half*)(sc + ROWS);                                    // [KSTAGE ? 32*(HD+8) : 0]
    const int head0=blockIdx.x*G, warp=threadIdx.x>>5, lane=threadIdx.x&31, tix=threadIdx.x;
    const int gid=lane>>2, tid=lane&3, kvh=head0/gqa, pos0=*d_pos;
    const int qbase=blockIdx.y*32, qn=(B-qbase<32)?(B-qbase):32;   // this CTA's 32-query tile (blockIdx.y); B>32 tiles over y
    float acc[2*G][FHDW/8][4];
    #pragma unroll
    for(int m=0;m<2*G;m++)for(int n=0;n<FHDW/8;n++)for(int c=0;c<4;c++)acc[m][n][c]=0.0f;
    for(int Q=tix;Q<ROWS;Q+=256){ sm[Q]=-1e30f; sl[Q]=0.0f; }
    for(int t=tix;t<ROWS*HD;t+=256){ int b=t/HD,i=t%HD;
        sQ[t]=((b&31)<qn)?__float2half(q[(size_t)(qbase+(b&31))*n_head*HD + (size_t)(head0+(b>>5))*HD + i]):(__half)0; }
    __syncthreads();
    int pmax=pos0+qbase+qn-1, smin=(window>0 && pos0+qbase-window+1>0)?pos0+qbase-window+1:0;
    int khi = (seg && bidir_hi > pmax) ? bidir_hi : pmax;   // media chunk: reach past this tile to the span end
    // f32 K at hd 256 (the SWA rings): stage each 32-key block into shared as
    // f16, once, cooperatively. Reading K per-fragment from global made the
    // f32->f16 pack the kernel's top stall by far (ncu: F2FP.PACK_AB eating
    // the LDG latency, long_scoreboard 11.5 cyc/instr) and read every K
    // element twice (both query sub-tiles). The stage is one coalesced sweep
    // with 32 independent loads per thread; per-element conversion is
    // unchanged -> byte-identical. +8 half pad per row spreads the fragment
    // reads off the 32-bank multiple. f16 K (global layers) keeps the direct
    // loads (no conversion to hide); the hd-512 f32 instantiations skip it
    // too (their sQ alone is 32 KB).
    for(int kb0=(smin/FKB)*FKB; kb0<=khi; kb0+=FKB){
        if(KSTAGE){
            for(int e=tix;e<32*HD;e+=256){
                int key=kb0+e/HD, rr=RING?key%seq:key;
                sK[(e/HD)*(HD+8)+e%HD]=__float2half(((const float*)Kc)[(size_t)rr*kv_dim+(size_t)kvh*HD+e%HD]);
            }
            __syncthreads();
        }
        int qt=warp>>2, kt=warp&3;
        float s[G][4];
        #pragma unroll
        for(int g2=0;g2<G;g2++){ s[g2][0]=0; s[g2][1]=0; s[g2][2]=0; s[g2][3]=0; }
        #pragma unroll
        for(int ks=0;ks<HD;ks+=16){
            uint32_t b0, b1;
            if(KSTAGE){
                const __half *kh=&sK[(kt*8+gid)*(HD+8)+ks];
                b0=*(const uint32_t*)(kh+tid*2); b1=*(const uint32_t*)(kh+tid*2+8);
            }else{
                int key=kb0+kt*8+gid, rr=RING?key%seq:key;
                const KT *kp=Kc+(size_t)rr*kv_dim+(size_t)kvh*HD+ks;
                b0=fa_ld2<KT>(kp+tid*2); b1=fa_ld2<KT>(kp+tid*2+8);
            }
            #pragma unroll
            for(int g2=0;g2<G;g2++){                       // one K fragment feeds all G heads' q-subtiles
                int qs=qt+2*g2;
                uint32_t a0=*(uint32_t*)&sQ[(qs*16+gid)*HD+ks+tid*2],   a1=*(uint32_t*)&sQ[(qs*16+gid+8)*HD+ks+tid*2];
                uint32_t a2=*(uint32_t*)&sQ[(qs*16+gid)*HD+ks+tid*2+8], a3=*(uint32_t*)&sQ[(qs*16+gid+8)*HD+ks+tid*2+8];
                fa_mma(s[g2][0],s[g2][1],s[g2][2],s[g2][3],a0,a1,a2,a3,b0,b1);
            }
        }
        #pragma unroll
        for(int g2=0;g2<G;g2++){
            int qs=qt+2*g2;
            sS[(qs*16+gid)*FKB+kt*8+tid*2]=s[g2][0];     sS[(qs*16+gid)*FKB+kt*8+tid*2+1]=s[g2][1];
            sS[(qs*16+gid+8)*FKB+kt*8+tid*2]=s[g2][2];   sS[(qs*16+gid+8)*FKB+kt*8+tid*2+1]=s[g2][3];
        }
        __syncthreads();
        #pragma unroll
        for(int r=0;r<4*G;r++){
            int b=warp*4*G+r, pos=pos0+qbase+(b&31), start=(window>0&&pos-window+1>0)?pos-window+1:0;
            int kabs=kb0+lane; bool ok=(lane<FKB)&&((b&31)<qn)&&(kabs>=start)&&(kabs<=pos);
            if(seg && !ok && lane<FKB && (b&31)<qn && kabs<=bidir_hi){   // same media span: bidirectional (bypasses causal+window)
                int sq=seg[pos]; ok=(sq!=0)&&(sq==seg[kabs]); }
            float s2=ok?sS[b*FKB+lane]:-1e30f, rmax=s2;
            for(int o=16;o>0;o>>=1) rmax=fmaxf(rmax,__shfl_xor_sync(~0u,rmax,o));
            float mn=fmaxf(sm[b],rmax), corr=__expf(sm[b]-mn), p=ok?__expf(s2-mn):0.0f, rsum=p;
            for(int o=16;o>0;o>>=1) rsum+=__shfl_xor_sync(~0u,rsum,o);
            if(lane<FKB) sP[b*FKB+lane]=__float2half(p);
            if(lane==0){ sl[b]=sl[b]*corr+rsum; sm[b]=mn; sc[b]=corr; }
        }
        __syncthreads();
        #pragma unroll
        for(int m=0;m<2*G;m++){ float cA=sc[m*16+gid], cB=sc[m*16+gid+8];
            for(int n=0;n<FHDW/8;n++){ acc[m][n][0]*=cA; acc[m][n][1]*=cA; acc[m][n][2]*=cB; acc[m][n][3]*=cB; } }
        // PV: V is query-INDEPENDENT, so read each V fragment ONCE and feed all
        // 2*G query-tiles (m) — the strided V traffic per head halves at G=2.
        #pragma unroll
        for(int ks2=0;ks2<FKB;ks2+=16){
            uint32_t pa[2*G][4];
            #pragma unroll
            for(int m=0;m<2*G;m++){
                pa[m][0]=*(uint32_t*)&sP[(m*16+gid)*FKB+ks2+tid*2];   pa[m][1]=*(uint32_t*)&sP[(m*16+gid+8)*FKB+ks2+tid*2];
                pa[m][2]=*(uint32_t*)&sP[(m*16+gid)*FKB+ks2+tid*2+8]; pa[m][3]=*(uint32_t*)&sP[(m*16+gid+8)*FKB+ks2+tid*2+8];
            }
            int k0=kb0+ks2+tid*2;
            int r0=RING?k0%seq:k0, r1=RING?(k0+1)%seq:k0+1, r8=RING?(k0+8)%seq:k0+8, r9=RING?(k0+9)%seq:k0+9;
            #pragma unroll
            for(int n=0;n<FHDW/8;n++){ int hdn=warp*FHDW+n*8+gid;
                const KT *vb=Vc+(size_t)kvh*HD+hdn;
                uint32_t b0=fa_pk(fa_rd1<KT>(vb+(size_t)r0*kv_dim), fa_rd1<KT>(vb+(size_t)r1*kv_dim));
                uint32_t b1=fa_pk(fa_rd1<KT>(vb+(size_t)r8*kv_dim), fa_rd1<KT>(vb+(size_t)r9*kv_dim));
                #pragma unroll
                for(int m=0;m<2*G;m++)
                    fa_mma(acc[m][n][0],acc[m][n][1],acc[m][n][2],acc[m][n][3],pa[m][0],pa[m][1],pa[m][2],pa[m][3],b0,b1);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int m=0;m<2*G;m++) for(int n=0;n<FHDW/8;n++){
        int rA=m*16+gid, rB=m*16+gid+8, hdc=warp*FHDW+n*8+tid*2;
        int hA=head0+(rA>>5), qA=rA&31, qB=rB&31;          // rA and rB sit in the same 32-row head half
        if(qA<qn){ xb[((size_t)(qbase+qA)*n_head+hA)*HD+hdc]=acc[m][n][0]/sl[rA]; xb[((size_t)(qbase+qA)*n_head+hA)*HD+hdc+1]=acc[m][n][1]/sl[rA]; }
        if(qB<qn){ xb[((size_t)(qbase+qB)*n_head+hA)*HD+hdc]=acc[m][n][2]/sl[rB]; xb[((size_t)(qbase+qB)*n_head+hA)*HD+hdc+1]=acc[m][n][3]/sl[rB]; }
    }
}
// dynamic-shared footprint of one flash CTA (must match the carve-up above)
static size_t flash_shm(int hd, int G, bool kstage) {
    return (size_t)32*G*hd*2 + (size_t)32*G*32*6 + (size_t)32*G*12 + (kstage ? (size_t)32*(hd+8)*2 : 0);
}
// The G=2 launches live in a specialization so the hd-512 packed instantiation
// (128 acc floats/thread — it would spill, and nothing ever launches it) never
// exists in the binary.
template<int HD> struct flash_packed {
    static bool go(float*, const float*, const void*, const void*, int, int, const int*, int, int, int, int,
                   bool, bool, const int*, int) { return false; }
};
template<> struct flash_packed<256> {
    static bool go(float *dxb, const float *dq, const void *Kc, const void *Vc,
                   int kv_dim, int gqa, const int *d_pos, int window, int seq, int B, int n_head,
                   bool f16, bool ring, const int *seg, int bidir_hi) {
        dim3 g(n_head/2, (B + 31) / 32);
        size_t shm = flash_shm(256, 2, !f16);
        static int carve = 0;
        if (!carve) {                                       // ~62 KB > the 48 KB default
            cudaFuncSetAttribute(flash_attn_n_kernel<256,false,__half,2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)flash_shm(256,2,false));
            cudaFuncSetAttribute(flash_attn_n_kernel<256,true, __half,2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)flash_shm(256,2,false));
            cudaFuncSetAttribute(flash_attn_n_kernel<256,true, float, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)flash_shm(256,2,true));
            cudaFuncSetAttribute(flash_attn_n_kernel<256,false,float, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)flash_shm(256,2,true));
            carve = 1;
        }
        if (f16 && ring) flash_attn_n_kernel<256,true, __half,2><<<g,256,shm>>>(dxb,dq,(const __half*)Kc,(const __half*)Vc,kv_dim,gqa,d_pos,window,seq,B,n_head,seg,bidir_hi);
        else if (f16)  flash_attn_n_kernel<256,false,__half,2><<<g,256,shm>>>(dxb,dq,(const __half*)Kc,(const __half*)Vc,kv_dim,gqa,d_pos,window,seq,B,n_head,seg,bidir_hi);
        else if (ring) flash_attn_n_kernel<256,true, float, 2><<<g,256,shm>>>(dxb,dq,(const float*)Kc,(const float*)Vc,kv_dim,gqa,d_pos,window,seq,B,n_head,seg,bidir_hi);
        else          flash_attn_n_kernel<256,false,float, 2><<<g,256,shm>>>(dxb,dq,(const float*)Kc,(const float*)Vc,kv_dim,gqa,d_pos,window,seq,B,n_head,seg,bidir_hi);
        return true;
    }
};
// hd is a compile-time template (256 SWA / 512 global); pick KT/RING like the
// per-query dispatch: f16 -> global (no ring), f32+seq<max -> SWA ring, else
// full f32. GQA packing (G=2, hd 256, gqa 2) defaults ON for integrated GPUs —
// it halves the K/V instruction stream the Orin is throttled on, but costs the
// A5000 a CTA of occupancy; LG_FLASH_GQA=0/1 overrides.
template<int HD>
static void launch_flash(float *dxb, const float *dq, const void *Kc, const void *Vc,
                         int kv_dim, int gqa, const int *d_pos, int window, int seq, int B, int n_head,
                         bool f16, bool ring, const int *seg, int bidir_hi) {
    static int pack_pol = -2;
    if (pack_pol == -2) {
        const char *e = getenv("LG_FLASH_GQA");
        if (e) pack_pol = atoi(e);
        else { cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); pack_pol = p.integrated ? 1 : 0; }
    }
    if (pack_pol && HD == 256 && gqa == 2 && (n_head & 1) == 0 &&
        flash_packed<HD>::go(dxb, dq, Kc, Vc, kv_dim, gqa, d_pos, window, seq, B, n_head, f16, ring, seg, bidir_hi))
        return;
    dim3 g(n_head, (B + 31) / 32);                         // y = 32-query tiles (B>32 prefill chunks)
    size_t shm = flash_shm(HD, 1, HD == 256 && !f16);
    if (f16 && ring) flash_attn_n_kernel<HD,true,__half,1><<<g,256,shm>>>(dxb,dq,(const __half*)Kc,(const __half*)Vc,kv_dim,gqa,d_pos,window,seq,B,n_head,seg,bidir_hi);
    else if (f16) flash_attn_n_kernel<HD,false,__half,1><<<g,256,shm>>>(dxb,dq,(const __half*)Kc,(const __half*)Vc,kv_dim,gqa,d_pos,window,seq,B,n_head,seg,bidir_hi);
    else if (ring) flash_attn_n_kernel<HD,true, float ,1><<<g,256,shm>>>(dxb,dq,(const float*)Kc,(const float*)Vc,kv_dim,gqa,d_pos,window,seq,B,n_head,seg,bidir_hi);
    else          flash_attn_n_kernel<HD,false,float ,1><<<g,256,shm>>>(dxb,dq,(const float*)Kc,(const float*)Vc,kv_dim,gqa,d_pos,window,seq,B,n_head,seg,bidir_hi);
}

// One prefill chunk's layer loop: the forward for the PREFILL_B positions
// whose inputs already sit in dx (and whose PLE rows, if the model has them,
// sit in d_ipl), at positions *d_pos..*d_pos+B-1, head skipped. The math is
// exactly B single-token forwards — every kernel above carries a row dimension
// that decode launches at 1 — but each weight matrix crosses the memory bus
// ONCE per chunk instead of once per token, and prefill is bandwidth-bound,
// so that factor is most of its cost. Per layer, the whole chunk's k/v are
// written to the cache before its queries run; causality holds because each
// query reads only up to its own position. Runs un-captured (and without the
// fork/join forks): a chunk already amortizes launch latency over its B tokens.
// LG_PREFILL_PROFILE=1: per-stage wall time across a whole prefill, with a
// sync after each stage group (slows the run; for attribution only).
static double g_pf_mm = 0, g_pf_attn = 0, g_pf_elem = 0, g_pf_ple = 0;
static int g_pf_on = -1;
static int pf_on(void) {
    if (g_pf_on < 0) g_pf_on = getenv("LG_PREFILL_PROFILE") != NULL;
    return g_pf_on;
}
static double pf_tick(double *acc) {
    static double last;
    if (!pf_on()) return 0;
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    double now;
    { struct timespec ts; timespec_get(&ts, TIME_UTC); now = (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }
    if (acc) *acc += now - last;
    last = now;
    return now;
}

static void chunk_layers(struct model *m, struct kvcache *kv, int has_ple, int B, matmul_fn mm) {
    const struct config *c = &m->cfg;
    const int n_embd = c->n_embd, n_head = c->n_head;
    const float eps = c->rms_eps;
    const float *d_rope_freqs = dW(m, "rope_freqs.weight");

    pf_tick(NULL);
    for (int L = 0; L < c->n_layer; L++) {
        const int local = m->is_local[L];
        const int hd = m->head_dim[L], n_head_kv = m->n_head_kv[L];
        const int q_dim = n_head * hd, kv_dim = n_head_kv * hd;
        const float base = local ? c->rope_freq_base_swa : c->rope_freq_base;
        const float *ff = local ? NULL : d_rope_freqs;
        const float *os = dW_layer(m, L, "layer_output_scale.weight");

        // ---- attention ----
        norm_n(dh, dx, dW_layer(m, L, "attn_norm.weight"), n_embd, eps, actq_for(B * n_embd), B);
        pf_tick(&g_pf_elem);

        int src = kv_src_dev(m, L);
        const int has_kv = L < c->n_kv_start;
        if (has_kv) {
            mm(dkb, wq_layer(m, L, "attn_k.weight"), dh, n_embd, kv_dim);
            const struct gguf_tensor *wv = wq_layer(m, L, "attn_v.weight");
            if (wv) mm(dvb, wv, dh, n_embd, kv_dim);
            else    CUDA_CHECK(cudaMemcpyAsync(dvb, dkb, (size_t)B * kv_dim * 4, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        }
        mm(dq, wq_layer(m, L, "attn_q.weight"), dh, n_embd, q_dim);
        pf_tick(&g_pf_mm);
        rmsnorm_kernel<<<B * n_head, 256>>>(dq, dq, dW_layer(m, L, "attn_q_norm.weight"), hd, eps, AQ0);
        rope_n_kernel<<<gridn(B * n_head * hd / 2), 256>>>(dq, hd / 2, hd, d_pos, rope_tab(base, hd), ff, n_head * hd / 2, B);
        if (has_kv) {
            rmsnorm_kernel<<<B * n_head_kv, 256>>>(dkb, dkb, dW_layer(m, L, "attn_k_norm.weight"), hd, eps, AQ0);
            rmsnorm_kernel<<<B * n_head_kv, 256>>>(dvb, dvb, NULL, hd, eps, AQ0);
            rope_n_kernel<<<gridn(B * n_head_kv * hd / 2), 256>>>(dkb, hd / 2, hd, d_pos, rope_tab(base, hd), ff, n_head_kv * hd / 2, B);
            if (kv->f16[L] && kv->seq[L] < kv->max_seq) {  // f16 ring (SWA layer)
                kv_write_ring_h_n_kernel<<<gridn(B * kv_dim), 256>>>((__half *)kv->k[L], dkb, d_pos, kv_dim, kv->seq[L], B);
                kv_write_ring_h_n_kernel<<<gridn(B * kv_dim), 256>>>((__half *)kv->v[L], dvb, d_pos, kv_dim, kv->seq[L], B);
            } else if (kv->f16[L]) {                       // f16 full-length: row = position
                kv_write_h_n_kernel<<<gridn(B * kv_dim), 256>>>((__half *)kv->k[L], dkb, d_pos, kv_dim, B);
                kv_write_h_n_kernel<<<gridn(B * kv_dim), 256>>>((__half *)kv->v[L], dvb, d_pos, kv_dim, B);
            } else if (kv->seq[L] < kv->max_seq) {         // f32 ring (LG_SWA_F32)
                kv_write_ring_n_kernel<<<gridn(B * kv_dim), 256>>>((float *)kv->k[L], dkb, d_pos, kv_dim, kv->seq[L], B);
                kv_write_ring_n_kernel<<<gridn(B * kv_dim), 256>>>((float *)kv->v[L], dvb, d_pos, kv_dim, kv->seq[L], B);
            } else {                                       // full-length f32: row = position
                kv_write_n_kernel<<<gridn(B * kv_dim), 256>>>((float *)kv->k[L], dkb, d_pos, kv_dim, B);
                kv_write_n_kernel<<<gridn(B * kv_dim), 256>>>((float *)kv->v[L], dvb, d_pos, kv_dim, B);
            }
        }
        const void *Kc = kv->k[src], *Vc = kv->v[src];
        pf_tick(&g_pf_elem);
        int gqa = n_head / n_head_kv;
        int window = (local && c->sliding_window > 0) ? c->sliding_window : 0;
        // K/V sharing pays only when the attended cache is too big to stay
        // L2-resident — then the per-query kernel's B× re-reads hit DRAM and
        // sharing them across the chunk's queries wins; when it fits L2 the
        // re-reads are already cheap and the sharing kernel's per-position
        // barriers only add overhead. The footprint is the K+V of the attended
        // span: full layers (window 0) grow unbounded -> always share; SWA
        // layers cap at `window` rows, so share only when window*kv_dim*KT is
        // big (true on 12B's wide kv_dim, false on E4B's narrow window). This
        // split is exactly what the Orin measured: sharing sped 12B's full AND
        // window layers but slowed E4B's window layers. The B<=2 MTP verify
        // always takes the per-query path (its argmax must match decode's).
        static long g_l2 = -1;
        if (g_l2 < 0) { cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); g_l2 = p.l2CacheSize; }
        int ktsz = kv->f16[src] ? 2 : 4;
        // Tensor-core flash for the prefill chunk (B>2): ~40% of TTFT was scalar-
        // dot attention. hd 256/512 only (Gemma-4 head dims); B<=2 MTP verify and
        // other hd keep the per-query path. LG_NO_FLASH falls back. Flash writes
        // f32 xb -> act_quantize fills the int8 activation for attn_output.
        static int no_flash = -1;
        if (no_flash < 0) no_flash = getenv("LG_NO_FLASH") != NULL;
        static int no_splitk = -1;
        if (no_splitk < 0) no_splitk = getenv("LG_NO_SPLITK") != NULL;
        // LG_FORCE_KVSHARE: route E4B's flash path through K/V-sharing instead. The
        // ncu showed flash is L1-bandwidth bound (92-95%) re-reading K/V per query;
        // the L2-footprint gate that disabled sharing on E4B measured the wrong
        // resource. Test whether staging K/V to shared (reuse across QT queries)
        // relieves the L1 wall. Text-prefill only (share path has no bidir mask).
        static int force_share = -1;
        if (force_share < 0) force_share = getenv("LG_FORCE_KVSHARE") != NULL;
        bool flash = !no_flash && !force_share && B > 2 && !g_chunk_verify && (hd == 256 || hd == 512);   // y-tiled over 32-query blocks, so B>32 (128) is fine
        bool share = (B > 2) && !g_chunk_verify && (force_share || window == 0 || 2LL * window * kv_dim * ktsz > (long)g_l2);
        // The B<=2 MTP verify runs the SAME split-K kernel as decode (one extra
        // grid axis for the 2 query rows): out[0]/out[1] then byte-match plain
        // decode's forwards at pos/pos+1 by construction, and verify's attention
        // gets the split-K parallelism win at high context. Per-query stays as the
        // LG_NO_SPLITK fallback (and there decode is per-query too, so they still
        // match). Mirrors decode's gate (decode-side LG_NO_SPLITK is independent).
        bool splitk = !no_splitk && (B <= 2 || g_chunk_verify) && (hd == 256 || hd == 512);
        if (flash) {
            bool f16 = kv->f16[src], ring = (kv->seq[src] < kv->max_seq);
            if (hd == 512) launch_flash<512>(dxb, dq, Kc, Vc, kv_dim, gqa, d_pos, window, kv->seq[src], B, n_head, f16, ring, g_pf_seg, g_pf_bidir_hi);
            else           launch_flash<256>(dxb, dq, Kc, Vc, kv_dim, gqa, d_pos, window, kv->seq[src], B, n_head, f16, ring, g_pf_seg, g_pf_bidir_hi);
            act_quantize_n(dxb, B * q_dim);
        } else if (share) {                                // attended K/V exceeds L2: share across queries
            const int QT = 8;                              // queries per block (one warp each)
            size_t shm = (size_t)(2 + QT) * hd * sizeof(float);   // sK + sV + sO[QT][hd]
            dim3 g(n_head, (B + QT - 1) / QT);
            struct actq aq = actq_for(B * q_dim);
            int ring = kv->seq[src] < kv->max_seq;
            if (kv->f16[src] && ring)                      // f16 ring (SWA layer)
                attn_kvshare_n_kernel<QT, true, __half><<<g, QT * 32, shm>>>(dxb, dq, (const __half *)Kc, (const __half *)Vc, hd, kv_dim, gqa, d_pos, window, kv->seq[src], B, n_head, aq);
            else if (kv->f16[src])                         // f16 full-length
                attn_kvshare_n_kernel<QT, false, __half><<<g, QT * 32, shm>>>(dxb, dq, (const __half *)Kc, (const __half *)Vc, hd, kv_dim, gqa, d_pos, window, 0, B, n_head, aq);
            else if (ring)                                 // f32 ring (LG_SWA_F32)
                attn_kvshare_n_kernel<QT, true, float><<<g, QT * 32, shm>>>(dxb, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, kv->seq[src], B, n_head, aq);
            else                                           // full-length float cache
                attn_kvshare_n_kernel<QT, false, float><<<g, QT * 32, shm>>>(dxb, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, 0, B, n_head, aq);
        } else if (splitk) {                               // B<=2 MTP verify: decode's split-K kernel, z=B queries
            ensure_split(B * n_head);                       // no-op (ensure_weights pre-allocated 2*n_head pre-capture)
            size_t shm = (size_t)(256 / 32) * hd * sizeof(float);
            dim3 gs(n_head, MAXSPLIT, B);
            struct actq aq = actq_for(B * q_dim);
            int ring = kv->seq[src] < kv->max_seq;
            if (kv->f16[src] && ring)
                split_attn_kernel<true, __half><<<gs, 256, shm>>>(g_pacc, g_pml, dq, (const __half *)Kc, (const __half *)Vc, hd, kv_dim, gqa, d_pos, window, kv->seq[src], n_head);
            else if (kv->f16[src])
                split_attn_kernel<false, __half><<<gs, 256, shm>>>(g_pacc, g_pml, dq, (const __half *)Kc, (const __half *)Vc, hd, kv_dim, gqa, d_pos, window, 0, n_head);
            else if (ring)
                split_attn_kernel<true, float><<<gs, 256, shm>>>(g_pacc, g_pml, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, kv->seq[src], n_head);
            else
                split_attn_kernel<false, float><<<gs, 256, shm>>>(g_pacc, g_pml, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, 0, n_head);
            combine_attn_kernel<<<dim3(n_head, 1, B), 256>>>(dxb, g_pacc, g_pml, hd, d_pos, window, aq, n_head);
        } else {
            size_t shm = (size_t)(256 / 32) * hd * sizeof(float);
            int ring = kv->seq[src] < kv->max_seq;
            if (kv->f16[src] && ring)
                attn_swa_h_n_kernel<<<dim3(n_head, B), 256, shm>>>(dxb, dq, (const __half *)Kc, (const __half *)Vc, hd, kv_dim, gqa, d_pos, window, kv->seq[src], actq_for(B * q_dim));
            else if (kv->f16[src])
                attn_h_n_kernel<<<dim3(n_head, B), 256, shm>>>(dxb, dq, (const __half *)Kc, (const __half *)Vc, hd, kv_dim, gqa, d_pos, window, actq_for(B * q_dim));
            else if (ring)
                attn_swa_n_kernel<<<dim3(n_head, B), 256, shm>>>(dxb, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, kv->seq[src], actq_for(B * q_dim));
            else
                attn_n_kernel<<<dim3(n_head, B), 256, shm>>>(dxb, dq, (const float *)Kc, (const float *)Vc, hd, kv_dim, gqa, d_pos, window, actq_for(B * q_dim));
        }

        pf_tick(&g_pf_attn);
        mm(dout, wq_layer(m, L, "attn_output.weight"), dxb, q_dim, n_embd);
        pf_tick(&g_pf_mm);
        norm_add_n(dx, dout, dW_layer(m, L, "post_attention_norm.weight"), n_embd, eps, NULL, AQ0, B);

        // ---- feed-forward (GeGLU) ----
        const int nff = m->ffn_len[L];
        norm_n(dh, dx, dW_layer(m, L, "ffn_norm.weight"), n_embd, eps, actq_for(B * n_embd), B);
        pf_tick(&g_pf_elem);
        mm(dg2, wq_layer(m, L, "ffn_up.weight"), dh, n_embd, nff);
        mm(dg1, wq_layer(m, L, "ffn_gate.weight"), dh, n_embd, nff);
        pf_tick(&g_pf_mm);
        geglu_n_kernel<<<gridn(B * nff), 256>>>(dg1, dg2, nff, B, nff, actq_for(B * nff));
        pf_tick(&g_pf_elem);
        mm(dout, wq_layer(m, L, "ffn_down.weight"), dg1, nff, n_embd);
        pf_tick(&g_pf_mm);
        norm_add_n(dx, dout, dW_layer(m, L, "post_ffw_norm.weight"), n_embd, eps,
                   has_ple ? NULL : os, has_ple ? actq_for(B * n_embd) : AQ0, B);
        pf_tick(&g_pf_elem);

        // ---- per-layer input (PLE) ----
        if (has_ple) {
            const int ple = c->n_embd_per_layer;
            mm(dpg, wq_layer(m, L, "inp_gate.weight"), dx, n_embd, ple);
            geglu_n_kernel<<<gridn(B * ple), 256>>>(dpg, d_ipl + (size_t)L * ple, ple, B, c->n_layer * ple, actq_for(B * ple));
            mm(dout, wq_layer(m, L, "proj.weight"), dpg, ple, n_embd);
            norm_add_n(dx, dout, dW_layer(m, L, "post_norm.weight"), n_embd, eps, os, AQ0, B);
            pf_tick(&g_pf_ple);
        }
    }
}

// Token form: look up and scale the chunk's embedding rows, build the PLE
// inputs, then run the layers.
static void forward_chunk(struct model *m, struct kvcache *kv, const int *tokens, int pos0, int cols) {
    const struct config *c = &m->cfg;
    g_pf_cols = cols;
    const int B = cols, n_embd = c->n_embd;

    float *rows = (float *)malloc((size_t)B * n_embd * 4);
    if (!rows) { fprintf(stderr, "forward_chunk: out of memory\n"); exit(1); }
    float es = sqrtf((float)n_embd);
    for (int j = 0; j < B; j++) {
        float *erow = dequantize_row(wq(m, "token_embd.weight"), tokens[j], n_embd);
        for (int i = 0; i < n_embd; i++) rows[(size_t)j * n_embd + i] = erow[i] * es;
        free(erow);
    }
    CUDA_CHECK(cudaMemcpy(dx, rows, (size_t)B * n_embd * 4, cudaMemcpyHostToDevice));
    free(rows);

    CUDA_CHECK(cudaMemcpy(d_pos, &pos0, sizeof(int), cudaMemcpyHostToDevice));  // kernels add the row index
    build_per_layer_n(m, tokens, B, matmul_q_n);
    chunk_layers(m, kv, model_has_ple(m), B, matmul_q_n);
}

// Embedding form (media tokens): the rows enter exactly as given — media
// embeddings are NOT sqrt(n_embd)-scaled (only real token lookups are). On a
// PLE model a media position takes the PADDING token's (id 0) per-layer row
// beside the usual projection of its embedding — the reference does exactly
// this for embedding batches; the 12B has no PLE at all.
static void forward_chunk_embd(struct model *m, struct kvcache *kv, const float *rows, int pos0, int cols) {
    const struct config *c = &m->cfg;
    g_pf_cols = cols;
    CUDA_CHECK(cudaMemcpy(dx, rows, (size_t)cols * c->n_embd * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos, &pos0, sizeof(int), cudaMemcpyHostToDevice));
    int has_ple = model_has_ple(m);
    if (has_ple) {
        int pad[PREFILL_MAX_B] = { 0 };
        build_per_layer_n(m, pad, cols, matmul_q_n);
    }
    chunk_layers(m, kv, has_ple, cols, matmul_q_n);
}

// Mixed form: each chunk position is a text token (ids[j] >= 0: looked-up,
// scaled embedding + its own per-layer row) or a media row (ids[j] < 0: row
// -ids[j]-1 of mrows as given + the padding token's per-layer row). Every
// position's math is exactly its pure-form chunk's — only the company a
// position keeps in a chunk changes.
static void forward_chunk_mixed(struct model *m, struct kvcache *kv, const float *mrows,
                                const int *ids, int pos0, int cols) {
    const struct config *c = &m->cfg;
    g_pf_cols = cols;
    const int B = cols, n_embd = c->n_embd;

    float *rows = (float *)malloc((size_t)B * n_embd * 4);
    if (!rows) { fprintf(stderr, "forward_chunk_mixed: out of memory\n"); exit(1); }
    float es = sqrtf((float)n_embd);
    int toks[PREFILL_MAX_B];
    for (int j = 0; j < B; j++) {
        if (ids[j] >= 0) {
            toks[j] = ids[j];
            float *erow = dequantize_row(wq(m, "token_embd.weight"), ids[j], n_embd);
            for (int i = 0; i < n_embd; i++) rows[(size_t)j * n_embd + i] = erow[i] * es;
            free(erow);
        } else {
            toks[j] = 0;
            memcpy(rows + (size_t)j * n_embd, mrows + (size_t)(-ids[j] - 1) * n_embd, (size_t)n_embd * 4);
        }
    }
    CUDA_CHECK(cudaMemcpy(dx, rows, (size_t)B * n_embd * 4, cudaMemcpyHostToDevice));
    free(rows);

    CUDA_CHECK(cudaMemcpy(d_pos, &pos0, sizeof(int), cudaMemcpyHostToDevice));
    build_per_layer_n(m, toks, B, matmul_q_n);
    chunk_layers(m, kv, model_has_ple(m), B, matmul_q_n);
}

// Pre-size the int8 activation scratch to the 128-wide max BEFORE any chunk or
// the decode-graph capture. Adaptive chunks make a short first turn size g_xq
// small; a later wider turn would then realloc it — and the captured decode
// graph references g_xq, so a realloc would leave it dangling. One max-size call
// up front (no-op on the f32 backend) keeps the pointer stable for the session.
static void prefill_act_presize(struct model *m) {
    static int done = 0;
    if (done) return;
    actq_for((int)((size_t)g_prefill_max_b * m->cfg.n_ff));   // n_ff is the widest activation (ffn_down input)
    done = 1;
}

extern "C" void model_prefill(struct model *m, struct kvcache *kv, const int *tokens, int n, int pos0) {
    wide_chunk_init();                                // before ensure_scratch sizes buffers + ring
    ensure_weights(m);
    ensure_scratch(m);
    prefill_act_presize(m);
    const int CB = g_wide_chunk ? g_wide_chunk : PREFILL_B;
    // Balanced chunks (see model_prefill_mixed): ceil(n/CB) near-equal chunks,
    // each a multiple of 64 above 128 (matmul_q_n's fat single launch) or of 32
    // below. The LAST chunk PADS to its rounded width instead of falling back to
    // single-token forwards — a short serve turn is mostly tail, and every
    // single costs a full decode pass. The pad repeats the last real token at
    // the following positions; those kv rows are rewritten by whatever comes
    // next before anything can read them (every consumer writes position p
    // before its attention reads p, and reads stop at its own position). Real
    // rows' math is the chunk path's: byte-identical.
    int i = 0;
    while (n - i >= 2) {
        int rem = n - i, nch = (rem + CB - 1) / CB;
        int cols = (rem + nch - 1) / nch;
        cols = cols > 128 ? (cols + 63) / 64 * 64 : (cols + 31) / 32 * 32;
        if (cols > CB) cols = CB;
        int real = rem < cols ? rem : cols;
        if (pos0 + i + cols > kv->max_seq) break;      // no room for the pad: singles below
        if (real == cols)
            forward_chunk(m, kv, tokens + i, pos0 + i, cols);
        else {                                         // tail: pad with the last real token
            int padded[PREFILL_MAX_B];
            for (int j = 0; j < cols; j++) padded[j] = tokens[i + (j < real ? j : real - 1)];
            forward_chunk(m, kv, padded, pos0 + i, cols);
        }
        i += real;
        // The warmup tokens exist so ensure_act's one-time cudaMallocs precede
        // graph capture; one chunk does all of them (its activations are B×
        // decode's), so skip straight to capture — otherwise a chunk-aligned
        // prompt would push the two un-captured tokens and the capture itself
        // into the generation timer.
        if (g_graph_warmups < 2) g_graph_warmups = 2;
    }
    if (pf_on() && i > 0)
        fprintf(stderr, "prefill profile (%d chunk tokens): matmul %.2fs, attention %.2fs, elementwise %.2fs, ple %.2fs\n",
                i, g_pf_mm, g_pf_attn, g_pf_elem, g_pf_ple);
    if (pf_on() && i > 0) matmul_coverage_print();
    for (; i < n; i++)                                // 0-1 tokens (or no room): singles
        forward_token(m, kv, tokens[i], pos0 + i, 0);
}

extern "C" void model_prefill_embd(struct model *m, struct kvcache *kv, const float *rows, int n, int pos0) {
    ensure_weights(m);
    ensure_scratch(m);
    prefill_act_presize(m);
    const int n_embd = m->cfg.n_embd;
    int i = 0;
    for (; n - i >= PREFILL_B; i += PREFILL_B)
        forward_chunk_embd(m, kv, rows + (size_t)i * n_embd, pos0 + i, PREFILL_B);
    if (i > 0 && g_graph_warmups < 2) g_graph_warmups = 2;
    // pad the remainder to an adaptive-width chunk (roundup to 32, not 128); the
    // last row repeats; the pad rows' kv is overwritten before it can be read.
    int rem = n - i, cols = ((rem + 31) / 32) * 32;
    if (cols > PREFILL_B) cols = PREFILL_B;
    if (rem >= 2 && pos0 + i + cols <= kv->max_seq) {
        float *padded = (float *)malloc((size_t)cols * n_embd * 4);
        if (padded) {
            for (int j = 0; j < cols; j++)
                memcpy(padded + (size_t)j * n_embd,
                       rows + (size_t)(i + (j < rem ? j : rem - 1)) * n_embd, (size_t)n_embd * 4);
            forward_chunk_embd(m, kv, padded, pos0 + i, cols);
            free(padded);
            if (g_graph_warmups < 2) g_graph_warmups = 2;
            return;
        }
    }
    for (; i < n; i++) {                              // 0-1 rows (or no room): singles
        CUDA_CHECK(cudaMemcpy(dx, rows + (size_t)i * n_embd, (size_t)n_embd * 4, cudaMemcpyHostToDevice));
        int pos = pos0 + i;
        CUDA_CHECK(cudaMemcpy(d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice));
        if (model_has_ple(m)) build_per_layer(m, 0);  // padding token's PLE row
        forward_graph(m, kv, 0);
    }
}

extern "C" void model_prefill_mixed(struct model *m, struct kvcache *kv, const float *rows,
                                    const int *ids, int n, int pos0) {
    wide_chunk_init();                                // before ensure_scratch sizes buffers + ring
    ensure_weights(m);
    ensure_scratch(m);
    prefill_act_presize(m);
    const int n_embd = m->cfg.n_embd;
    // Text packing budget: LG_WIDE_CHUNK opens serve-path text to wide chunks too
    // (same knob and buffers as model_prefill).
    int TB = g_wide_chunk ? g_wide_chunk : PREFILL_B;
    if (TB > g_prefill_max_b) TB = g_prefill_max_b;

    // Per-position media-span ids: text -> 0, a media token -> its span's start abs
    // position+1 (a unique nonzero). Uploaded once so the flash mask can attend
    // bidirectionally within a frame (Gemma's image/video token-type behaviour).
    if (kv->max_seq > g_seg_cap) {
        if (g_seg_dev) cudaFree(g_seg_dev);
        CUDA_CHECK(cudaMalloc(&g_seg_dev, (size_t)kv->max_seq * sizeof(int)));
        CUDA_CHECK(cudaMemset(g_seg_dev, 0, (size_t)kv->max_seq * sizeof(int)));
        g_seg_cap = kv->max_seq;
    }
    int seg_ok = (pos0 + n <= kv->max_seq);
    if (seg_ok) {
        int *hseg = (int *)malloc((size_t)n * sizeof(int));
        int run0 = 0;
        for (int j = 0; j < n; j++) {
            if (ids[j] < 0) { if (j == 0 || ids[j - 1] >= 0) run0 = pos0 + j + 1; hseg[j] = run0; }
            else hseg[j] = 0;
        }
        CUDA_CHECK(cudaMemcpy(g_seg_dev + pos0, hseg, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
        free(hseg);
    }

    // Segment-aware chunking: pack text and WHOLE media spans greedily to the
    // balanced budget — a span never splits across chunks (a frame's patches
    // must coexist in one attention pass for bidirectional-within-frame), but
    // it packs WITH its neighbours: a camera turn's opener, its 130-row frame
    // and the question ride one chunk. (The old rule gave any span wider than
    // PREFILL_B its own chunk, so that same turn paid three padded launches —
    // 32+192+32 columns for ~157 real positions.) Text-only chunks pass
    // seg=NULL -> byte-identical causal; chunking matches the old 128 path for text.
    int i = 0;
    while (n - i >= 2) {
        int a = i, w = 0, media_hi = -1;
        // Balanced budget: split the remaining ids into ceil(rem/TB) NEAR-EQUAL
        // chunks instead of TB-wide chunks plus a skinny tail — a 1428-token turn
        // becomes 2 x ~714, not 1024 + 404. Same number of weight passes, but
        // every launch keeps the fat single-launch grid shape (and the tail
        // chunk stops being an under-filled straggler on a 64-SM card).
        int rem = n - i, nch = (rem + TB - 1) / TB;
        int budget = (rem + nch - 1) / nch;
        budget = budget > 128 ? (budget + 63) / 64 * 64 : (budget + 31) / 32 * 32;
        if (budget > TB) budget = TB;
        while (a + w < n && w < budget) {
            if (ids[a + w] < 0) {                      // a media span: take it whole if it fits
                int s = a + w, e = s; while (e < n && ids[e] < 0) e++;
                int span = e - s;
                if (w + span > budget) break;          // doesn't fit what's left -> close chunk
                media_hi = e - 1; w += span;
            } else w++;                                // text
        }
        // A span wider than the whole budget still prefills as ONE chunk,
        // stretched up to g_prefill_max_b (the buffer width); wider still and
        // it falls back to causal sub-chunks (Stage 2 widens the chunk for it).
        if (w == 0 && ids[a] < 0) {
            int e = a; while (e < n && ids[e] < 0) e++;
            int span = e - a;
            w = span <= g_prefill_max_b ? span : g_prefill_max_b;
            media_hi = a + w - 1;
        }
        if (w < 1) w = 1;
        // Chunks wider than 128 round to a multiple of 64: matmul_q_n's single
        // fat launch needs cols%64==0, and a few pad tokens beat falling back
        // to per-64-column launches for the whole chunk.
        int cols = w > 128 ? ((w + 63) / 64) * 64 : ((w + 31) / 32) * 32;
        if (cols > g_prefill_max_b) cols = g_prefill_max_b;
        if (pos0 + a + cols > kv->max_seq) break;
        int padded[PREFILL_MAX_B];
        for (int j = 0; j < cols; j++) padded[j] = ids[a + (j < w ? j : w - 1)];
        static int no_bidir = -1; if (no_bidir < 0) no_bidir = getenv("LG_NO_IMG_BIDIR") != NULL;
        int bidir = seg_ok && media_hi >= 0 && !no_bidir;
        g_pf_seg = bidir ? g_seg_dev : NULL;
        g_pf_bidir_hi = bidir ? (pos0 + media_hi) : 0;
        forward_chunk_mixed(m, kv, rows, padded, pos0 + a, cols);
        if (g_graph_warmups < 2) g_graph_warmups = 2;
        i = a + w;
    }
    g_pf_seg = NULL; g_pf_bidir_hi = 0;
    if (pf_on() && n >= 32)                           // cumulative across turns; diff successive lines
        fprintf(stderr, "prefill profile (mixed, +%d tokens, cum): matmul %.2fs, attention %.2fs, elementwise %.2fs, ple %.2fs\n",
                n, g_pf_mm, g_pf_attn, g_pf_elem, g_pf_ple);
    for (; i < n; i++) {                              // 0-1 trailing positions: singles
        if (ids[i] >= 0) { forward_token(m, kv, ids[i], pos0 + i, 0); continue; }
        CUDA_CHECK(cudaMemcpy(dx, rows + (size_t)(-ids[i] - 1) * n_embd, (size_t)n_embd * 4, cudaMemcpyHostToDevice));
        int pos = pos0 + i;
        CUDA_CHECK(cudaMemcpy(d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice));
        if (model_has_ple(m)) build_per_layer(m, 0);  // padding token's PLE row
        forward_graph(m, kv, 0);
    }
}


#endif // PREFILL_KERNEL_CUH
