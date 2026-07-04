// mtp-cuda.cuh — the device side of MTP speculative decoding, split from
// model-cuda.cuh (same single-include unit: model-cuda.cuh includes this at
// the point the code used to sit, so declaration order and codegen are
// unchanged). Two halves: the VERIFY (the LG_MTP_N-token block through the
// target's chunk path + head, captured as a static CUDA graph) and the DRAFT
// (the gemma4-assistant head resident on-device, its own capture). The host
// orchestration and the draft's reference implementation live in src/mtp.c.
#ifndef MTP_CUDA_CUH
#define MTP_CUDA_CUH

// The MTP verify step: the LG_MTP_N tokens (toks[0]=chosen, toks[1..]=drafts) as ONE
// B=LG_MTP_N chunk — each weight matrix crosses memory once for the whole block, the
// entire economics of speculative decoding on a bandwidth-bound device. The head runs
// at ALL rows (the only place that happens; prefill always skips it). out[j] = greedy
// successor of toks[j]; out[0] always valid, out[j] valid only when every earlier
// draft held. g_hidden is left at the last VALID row's post-norm hidden so the next
// draft chains on accept and reject alike; rejected rows are overwritten before any
// query reads them (causality, no rollback). Everything reads d_pos on-device, so node
// parameters never vary -> captures into a static CUDA graph like decode (the
// un-captured form is ~1030 raw launches/step, where sync MTP died on WDDM).
static void verify_head_spec(struct model *m) {
    const struct config *c = &m->cfg;
    const int n_embd = c->n_embd, N = LG_MTP_N;
    rmsnorm_kernel<<<N, NORM_THREADS(n_embd)>>>(dx, dx, dW(m, "output_norm.weight"), n_embd, c->rms_eps, actq_for(N * n_embd));
    matmul_q_spec(d_logits_spec, wq(m, "token_embd.weight"), dx, n_embd, c->n_vocab);
    if (c->logit_softcap > 0.0f)
        softcap_kernel<<<gridn(N * c->n_vocab), 256>>>(d_logits_spec, c->logit_softcap, N * c->n_vocab);
    for (int j = 0; j < N; j++)
        argmax_kernel<<<1, 1024>>>(d_logits_spec + (size_t)j * c->n_vocab, c->n_vocab, d_best_spec + j);
}
static void verify_layers_and_head_spec(struct model *m, struct kvcache *kv, int has_ple) {
    g_chunk_verify = 1;                  // verify uses decode's split-K attn, not prefill flash/share
    chunk_layers(m, kv, has_ple, LG_MTP_N, matmul_q_spec);
    g_chunk_verify = 0;
    verify_head_spec(m);
}

static double now_sec_dev(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// LG_MTP_PROFILE=1: run the verify un-captured with syncs around its two
// halves and report — the tool that finds where a 2x-over-theory round goes.
static void verify_profiled(struct model *m, struct kvcache *kv, int has_ple) {
    static double tl = 0, th = 0;
    static int n = 0;
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    double t0 = now_sec_dev();
    g_chunk_verify = 1;
    chunk_layers(m, kv, has_ple, LG_MTP_N, matmul_q_spec);
    g_chunk_verify = 0;
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    double t1 = now_sec_dev();
    verify_head_spec(m);
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    tl += t1 - t0; th += now_sec_dev() - t1;
    if (++n % 50 == 0)
        fprintf(stderr, "verify profile over %d: layers %.1fms ea, head %.1fms ea\n", n, 1e3 * tl / n, 1e3 * th / n);
}

static cudaGraphExec_t g_graph_spec_exec = NULL;
static int g_graph_spec_warmups = 0;
static void verify_graph_spec(struct model *m, struct kvcache *kv, int has_ple) {
    if (g_graph_spec_warmups < 2) {              // one-time mallocs (ensure_act) finish un-captured
        g_graph_spec_warmups++;
        verify_layers_and_head_spec(m, kv, has_ple);
        return;
    }
    if (!g_graph_spec_exec) {
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        verify_layers_and_head_spec(m, kv, has_ple);
        CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&g_graph_spec_exec, graph, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaGraphLaunch(g_graph_spec_exec, cudaStreamPerThread));
}

extern "C" int model_forward_spec(struct model *m, struct kvcache *kv, const int *toks, int pos, int *out) {
    const struct config *c = &m->cfg;
    const int n_embd = c->n_embd, N = LG_MTP_N;
    float *rows = (float *)malloc((size_t)N * n_embd * 4);
    if (!rows) { fprintf(stderr, "model_forward_spec: out of memory\n"); exit(1); }
    float es = sqrtf((float)n_embd);
    for (int j = 0; j < N; j++) {
        float *erow = dequantize_row(wq(m, "token_embd.weight"), toks[j], n_embd);
        for (int i = 0; i < n_embd; i++) rows[(size_t)j * n_embd + i] = erow[i] * es;
        free(erow);
    }
    CUDA_CHECK(cudaMemcpy(dx, rows, (size_t)N * n_embd * 4, cudaMemcpyHostToDevice));
    free(rows);
    CUDA_CHECK(cudaMemcpy(d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice));

    int has_ple = model_has_ple(m);
    if (has_ple) build_per_layer_n(m, toks, N, matmul_q_spec);   // un-captured, like decode's PLE build
    static int prof = -1;
    if (prof < 0) prof = getenv("LG_MTP_PROFILE") != NULL;
    if (prof) verify_profiled(m, kv, has_ple);
    else      verify_graph_spec(m, kv, has_ple);
    if (model_pick) {
        // Sample the target's own distribution at every row; a draft is accepted
        // only when the sample agrees, so the emitted tokens are exactly target
        // samples — speculation changes speed, never the distribution. (The
        // captured graph's argmaxes still run; their result just isn't read.)
        float *rows_h = pick_rows((size_t)N * c->n_vocab);
        CUDA_CHECK(cudaMemcpy(rows_h, d_logits_spec, (size_t)N * c->n_vocab * 4, cudaMemcpyDeviceToHost));
        for (int j = 0; j < N; j++) out[j] = model_pick(rows_h + (size_t)j * c->n_vocab, c->n_vocab);
    } else
        CUDA_CHECK(cudaMemcpy(out, d_best_spec, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));

    // accept run: out[0] always valid; out[j] valid iff every earlier draft held.
    int adv = 1;
    while (adv < N && out[adv - 1] == toks[adv]) adv++;
    CUDA_CHECK(cudaMemcpyAsync(g_hidden, dx + (size_t)(adv - 1) * n_embd, (size_t)n_embd * 4,
                               cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    return adv;
}

// ==== MTP: the draft head, on the device =====================================
// The gemma4-assistant forward from src/mtp.c as ~30 kernel launches per
// draft. Almost everything is REUSE: rmsnorm_kernel (pre/post norms and, with
// grid = heads, the per-head q norm — exactly how decode uses it),
// geglu_kernel, softcap_kernel, argmax_kernel, and above all the d_attn
// template: a draft for position `pos` sees the cache exactly as a query at
// pos-1 with the window shrunk by one — [pos-window+1, pos-1] sliding,
// [0, pos-1] full — so two NEW wrappers below pin those arguments and the
// frozen decode entry points stay untouched. Assistant weights are dequantized
// once to f16 on the device (77M params, ~150 MB); h_prev is g_hidden, which
// the decode graph already maintains. The whole draft CAPTURES into one CUDA
// graph (position read on-device from d_dpos, like decode's d_pos), so a
// round costs one graph launch + one 4-byte readback instead of ~30 raw
// launches — on WDDM that launch tax was most of the draft's wall time.

__global__ static void mtp_matvec_h(float *out, const __half *W, const float *x, int k, int m) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= m) return;
    const __half2 *w2 = (const __half2 *)(const void *)(W + (size_t)row * k);
    float s = 0.0f;
    for (int i = lane; i < k / 2; i += 32) {
        float2 wf = __half22float2(w2[i]);
        s += wf.x * x[2 * i] + wf.y * x[2 * i + 1];
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (!lane) out[row] = s;
}

// NeoX rope at the device-resident draft position (static graph node). No
// freq factors: the assistant has none.
__global__ static void mtp_rope(float *v, int half, int hd, const int *dp, float base, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int pos = *dp;
    int head = idx / half, i = idx % half;
    float *vh = v + (size_t)head * hd;
    float freq = powf(base, -2.0f * (float)i / (float)hd);
    float ang = (float)pos * freq;
    float c = cosf(ang), s = sinf(ang), a = vh[i], b = vh[i + half];
    vh[i] = a * c - b * s; vh[i + half] = a * s + b * c;
}

__global__ static void mtp_add_scale(float *x, const float *o, int n, float sc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (x[i] + o[i]) * sc;
}

// The draft at *dp sees the cache as a query at *dp - 1 (see the host draft).
__global__ static void mtp_attn_h(float *xb, const float *q, const __half *Kc, const __half *Vc,
                                  int hd, int kv_dim, int gqa, const int *dp, struct actq aq) {
    d_attn<false, __half>(xb, q, Kc, Vc, hd, kv_dim, gqa, *dp - 1, 0, 0, 0, aq);
}
__global__ static void mtp_attn_swa(float *xb, const float *q, const float *Kc, const float *Vc,
                                    int hd, int kv_dim, int gqa, const int *dp, int window, int seq, struct actq aq) {
    d_attn<true, float>(xb, q, Kc, Vc, hd, kv_dim, gqa, *dp - 1, 0, window, seq, aq);
}
__global__ static void mtp_attn_swa_h(float *xb, const float *q, const __half *Kc, const __half *Vc,
                                      int hd, int kv_dim, int gqa, const int *dp, int window, int seq, struct actq aq) {
    d_attn<true, __half>(xb, q, Kc, Vc, hd, kv_dim, gqa, *dp - 1, 0, window, seq, aq);
}

struct mtp_ld {
    float *attn_norm, *q_norm, *post_attn, *ffn_norm, *post_ffw;   // f32, device
    float out_scale;                                               // scalar, by value
    __half *q, *o, *gate, *up, *down;
};

struct mtp_cuda {
    struct mtp_ld *l;
    __half *pre, *post, *head;
    float *out_norm;
    float *cat, *x, *h, *q, *xb, *o, *g1, *g2, *logits;            // device scratch
    int *d_tok;
    int *d_dpos;                  // device-resident draft position (static graph)
    cudaGraphExec_t graph;
    int warmups;
};

static __half *mtp_up_h(const struct gguf_tensor *t) {            // any type -> device f16
    size_t n = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) n *= t->dims[i];
    float *f = (float *)malloc(n * 4);
    __half *hh = (__half *)malloc(n * 2);
    __half *d = NULL;
    int ok = f && hh && dequantize_into(t->type, t->data, f, (int64_t)n) &&
             cudaMalloc(&d, n * 2) == cudaSuccess;
    if (ok) {
        for (size_t i = 0; i < n; i++) hh[i] = __float2half(f[i]);
        ok = cudaMemcpy(d, hh, n * 2, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    free(f); free(hh);
    return ok ? d : NULL;
}

static float *mtp_up_f(const float *src, size_t n) {
    float *d = NULL;
    if (cudaMalloc(&d, n * 4) != cudaSuccess) return NULL;
    return cudaMemcpy(d, src, n * 4, cudaMemcpyHostToDevice) == cudaSuccess ? d : NULL;
}

static struct mtp_cuda *mtp_cuda_init(struct mtp *t) {
    struct mtp_cuda *mc = (struct mtp_cuda *)calloc(1, sizeof *mc);
    if (!mc) return NULL;
    mc->l = (struct mtp_ld *)calloc((size_t)t->n_layer, sizeof *mc->l);
    mc->pre  = mtp_up_h(t->pre);
    mc->head = mtp_up_h(t->head);
    mc->out_norm = mtp_up_f(t->out_norm, (size_t)t->n_inner);
    // post-projection (next-token), ni->nb: lifts the head's own hidden back to
    // model width to chain a 2nd draft (LG_MTP_N=3 block-3). Optional — left NULL
    // (block-3 chaining disabled) if it's absent or not the expected ni->nb shape.
    if (t->post && t->post->n_dims == 2 &&
        (int)t->post->dims[0] == t->n_inner && (int)t->post->dims[1] == t->n_bb)
        mc->post = mtp_up_h(t->post);
    int ok = mc->l && mc->pre && mc->head && mc->out_norm;
    int q_max = 0;
    for (int L = 0; ok && L < t->n_layer; L++) {
        const struct mtp_layer *s = &t->l[L];
        struct mtp_ld *d = &mc->l[L];
        if (t->n_head * s->hd > q_max) q_max = t->n_head * s->hd;
        d->out_scale = s->out_scale ? s->out_scale[0] : 1.0f;
        ok = (d->attn_norm = mtp_up_f(s->attn_norm, t->n_inner)) &&
             (d->q_norm = mtp_up_f(s->q_norm, s->hd)) &&
             (d->post_attn = mtp_up_f(s->post_attn, t->n_inner)) &&
             (d->ffn_norm = mtp_up_f(s->ffn_norm, t->n_inner)) &&
             (d->post_ffw = mtp_up_f(s->post_ffw, t->n_inner)) &&
             (d->q = mtp_up_h(s->q)) && (d->o = mtp_up_h(s->o)) &&
             (d->gate = mtp_up_h(s->gate)) && (d->up = mtp_up_h(s->up)) &&
             (d->down = mtp_up_h(s->down)) != NULL;
    }
    ok = ok && cudaMalloc(&mc->cat, (size_t)2 * t->n_bb * 4) == cudaSuccess
            && cudaMalloc(&mc->x, (size_t)t->n_inner * 4) == cudaSuccess
            && cudaMalloc(&mc->h, (size_t)t->n_inner * 4) == cudaSuccess
            && cudaMalloc(&mc->q, (size_t)q_max * 4) == cudaSuccess
            && cudaMalloc(&mc->xb, (size_t)q_max * 4) == cudaSuccess
            && cudaMalloc(&mc->o, (size_t)t->n_inner * 4) == cudaSuccess
            && cudaMalloc(&mc->g1, (size_t)t->n_ff * 4) == cudaSuccess
            && cudaMalloc(&mc->g2, (size_t)t->n_ff * 4) == cudaSuccess
            && cudaMalloc(&mc->logits, (size_t)t->n_vocab * 4) == cudaSuccess
            && cudaMalloc(&mc->d_tok, sizeof(int)) == cudaSuccess
            && cudaMalloc(&mc->d_dpos, sizeof(int)) == cudaSuccess;
    if (!ok) {                                       // startup-OOM only; partial uploads leak, like media-cuda
        fprintf(stderr, "mtp: device upload failed, drafting disabled\n");
        free(mc->l); free(mc);
        return NULL;
    }
    return mc;
}

// Every launch in the draft, position read from mc->d_dpos — capturable.
static void mtp_draft_launches(struct mtp *t, const struct model *m, const struct kvcache *kv,
                               struct mtp_cuda *mc) {
    const struct config *c = &m->cfg;
    const int ni = t->n_inner, nb = t->n_bb, nff = t->n_ff;
    const float eps = t->eps;

    // cat (= [sqrt-scaled embed(token), h_prev]) is filled by the caller BEFORE
    // this runs, so the captured graph is identical for draft 1 (h_prev=g_hidden)
    // and the chained draft 2 (h_prev=post(head hidden)).
    mtp_matvec_h<<<gridn(ni * 32), 256>>>(mc->x, mc->pre, mc->cat, 2 * nb, ni);

    for (int L = 0; L < t->n_layer; L++) {
        const struct mtp_layer *bl = &t->l[L];
        const struct mtp_ld *dl = &mc->l[L];
        const int hd = bl->hd, src = bl->src, q_dim = t->n_head * hd;
        const int gqa = t->n_head / m->n_head_kv[src];

        rmsnorm_kernel<<<1, NORM_THREADS(ni)>>>(mc->h, mc->x, dl->attn_norm, ni, eps, AQ0);
        mtp_matvec_h<<<gridn(q_dim * 32), 256>>>(mc->q, dl->q, mc->h, ni, q_dim);
        rmsnorm_kernel<<<t->n_head, 256>>>(mc->q, mc->q, dl->q_norm, hd, eps, AQ0);
        mtp_rope<<<gridn(t->n_head * (hd / 2)), 256>>>(mc->q, hd / 2, hd, mc->d_dpos,
                                                       bl->local ? t->base_swa : t->base_full,
                                                       t->n_head * (hd / 2));
        if (bl->local && kv->f16[src])                     // f16 ring (SWA layer)
            mtp_attn_swa_h<<<t->n_head, 256, 8 * hd * 4>>>(mc->xb, mc->q,
                    (const __half *)kv->k[src], (const __half *)kv->v[src],
                    hd, kv->kv_dim[src], gqa, mc->d_dpos,
                    c->sliding_window > 0 ? c->sliding_window - 1 : 0, kv->seq[src], AQ0);
        else if (bl->local)                                // f32 ring (LG_SWA_F32)
            mtp_attn_swa<<<t->n_head, 256, 8 * hd * 4>>>(mc->xb, mc->q,
                    (const float *)kv->k[src], (const float *)kv->v[src],
                    hd, kv->kv_dim[src], gqa, mc->d_dpos,
                    c->sliding_window > 0 ? c->sliding_window - 1 : 0, kv->seq[src], AQ0);
        else
            mtp_attn_h<<<t->n_head, 256, 8 * hd * 4>>>(mc->xb, mc->q,
                    (const __half *)kv->k[src], (const __half *)kv->v[src],
                    hd, kv->kv_dim[src], gqa, mc->d_dpos, AQ0);
        mtp_matvec_h<<<gridn(ni * 32), 256>>>(mc->o, dl->o, mc->xb, q_dim, ni);
        rmsnorm_kernel<<<1, NORM_THREADS(ni)>>>(mc->o, mc->o, dl->post_attn, ni, eps, AQ0);
        mtp_add_scale<<<gridn(ni), 256>>>(mc->x, mc->o, ni, 1.0f);

        rmsnorm_kernel<<<1, NORM_THREADS(ni)>>>(mc->h, mc->x, dl->ffn_norm, ni, eps, AQ0);
        mtp_matvec_h<<<gridn(nff * 32), 256>>>(mc->g1, dl->gate, mc->h, ni, nff);
        mtp_matvec_h<<<gridn(nff * 32), 256>>>(mc->g2, dl->up, mc->h, ni, nff);
        geglu_kernel<<<gridn(nff), 256>>>(mc->g1, mc->g2, nff, AQ0);
        mtp_matvec_h<<<gridn(ni * 32), 256>>>(mc->o, dl->down, mc->g1, nff, ni);
        rmsnorm_kernel<<<1, NORM_THREADS(ni)>>>(mc->o, mc->o, dl->post_ffw, ni, eps, AQ0);
        mtp_add_scale<<<gridn(ni), 256>>>(mc->x, mc->o, ni, dl->out_scale);
    }

    rmsnorm_kernel<<<1, NORM_THREADS(ni)>>>(mc->x, mc->x, mc->out_norm, ni, eps, AQ0);
    mtp_matvec_h<<<gridn(t->n_vocab * 32), 256>>>(mc->logits, mc->head, mc->x, ni, t->n_vocab);
    if (t->softcap > 0.0f)
        softcap_kernel<<<gridn(t->n_vocab), 256>>>(mc->logits, t->softcap, t->n_vocab);
    argmax_kernel<<<1, 1024>>>(mc->logits, t->n_vocab, mc->d_tok);
}

extern "C" int mtp_draft_device(struct mtp *t, const struct model *m, const struct kvcache *kv,
                                int token, int pos) {
    if (t->cuda == (void *)-1) return -1;
    struct mtp_cuda *mc = (struct mtp_cuda *)t->cuda;
    if (!mc) {
        mc = mtp_cuda_init(t);
        t->cuda = mc ? (void *)mc : (void *)-1;
        if (!mc) return -1;
    }
    const int nb = t->n_bb;

    // uploads stay outside the graph (fixed buffers, varying data)
    float *erow = dequantize_row(gguf_find_tensor(m->ctx, "token_embd.weight"), token, nb);
    if (!erow) return -1;
    float sc = sqrtf((float)nb);
    for (int i = 0; i < nb; i++) erow[i] *= sc;
    CUDA_CHECK(cudaMemcpy(mc->cat, erow, (size_t)nb * 4, cudaMemcpyHostToDevice));
    free(erow);
    CUDA_CHECK(cudaMemcpy(mc->d_dpos, &pos, sizeof(int), cudaMemcpyHostToDevice));
    // draft 1 chains on the TARGET's hidden (g_hidden); filled here, outside the
    // captured head-pass graph, so the chained draft 2 can reuse the same graph.
    CUDA_CHECK(cudaMemcpyAsync(mc->cat + nb, g_hidden, (size_t)nb * 4, cudaMemcpyDeviceToDevice, cudaStreamPerThread));

    if (mc->warmups < 2) {
        mc->warmups++;
        mtp_draft_launches(t, m, kv, mc);
    } else {
        if (!mc->graph) {
            cudaGraph_t graph;
            CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            mtp_draft_launches(t, m, kv, mc);
            CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            CUDA_CHECK(cudaGraphInstantiate(&mc->graph, graph, 0));
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
        CUDA_CHECK(cudaGraphLaunch(mc->graph, cudaStreamPerThread));
    }

    int best = -1;
    CUDA_CHECK(cudaMemcpy(&best, mc->d_tok, sizeof(int), cudaMemcpyDeviceToHost));
    return best;
}

// The chained draft for block-3 (LG_MTP_N=3): draft the token after `token` (itself
// the previous draft, sitting at `pos`-1) using the head's OWN hidden from the prior
// draft (mc->x, lifted ni->nb by the post-projection) as h_prev — the target never
// ran this position, so its hidden isn't available, which is why a chained draft is
// inherently weaker. Reuses draft 1's captured head-pass graph; cat is filled here
// before the launch. Returns -1 if the post-projection is unavailable (no chaining).
extern "C" int mtp_draft_chain_device(struct mtp *t, const struct model *m, const struct kvcache *kv,
                                      int token, int pos) {
    if (t->cuda == (void *)-1) return -1;
    struct mtp_cuda *mc = (struct mtp_cuda *)t->cuda;
    if (!mc || !mc->post) return -1;
    const int nb = t->n_bb, ni = t->n_inner;

    float *erow = dequantize_row(gguf_find_tensor(m->ctx, "token_embd.weight"), token, nb);
    if (!erow) return -1;
    float sc = sqrtf((float)nb);
    for (int i = 0; i < nb; i++) erow[i] *= sc;
    CUDA_CHECK(cudaMemcpy(mc->cat, erow, (size_t)nb * 4, cudaMemcpyHostToDevice));
    free(erow);
    // h_prev = post(previous draft's head hidden in mc->x); runs before the graph's
    // pre-projection overwrites mc->x (same stream -> ordered).
    mtp_matvec_h<<<gridn(nb * 32), 256>>>(mc->cat + nb, mc->post, mc->x, ni, nb);
    CUDA_CHECK(cudaMemcpy(mc->d_dpos, &pos, sizeof(int), cudaMemcpyHostToDevice));

    if (mc->graph) CUDA_CHECK(cudaGraphLaunch(mc->graph, cudaStreamPerThread));
    else           mtp_draft_launches(t, m, kv, mc);   // graph not captured yet (warmup rounds)

    int best = -1;
    CUDA_CHECK(cudaMemcpy(&best, mc->d_tok, sizeof(int), cudaMemcpyDeviceToHost));
    return best;
}

extern "C" void mtp_free_device(struct mtp *t) {
    struct mtp_cuda *mc = (struct mtp_cuda *)t->cuda;
    if (!mc || t->cuda == (void *)-1) return;
    for (int L = 0; L < t->n_layer; L++) {
        struct mtp_ld *d = &mc->l[L];
        cudaFree(d->attn_norm); cudaFree(d->q_norm); cudaFree(d->post_attn);
        cudaFree(d->ffn_norm); cudaFree(d->post_ffw);
        cudaFree(d->q); cudaFree(d->o); cudaFree(d->gate); cudaFree(d->up); cudaFree(d->down);
    }
    cudaFree(mc->pre); cudaFree(mc->post); cudaFree(mc->head); cudaFree(mc->out_norm);
    cudaFree(mc->cat); cudaFree(mc->x); cudaFree(mc->h); cudaFree(mc->q); cudaFree(mc->xb);
    cudaFree(mc->o); cudaFree(mc->g1); cudaFree(mc->g2); cudaFree(mc->logits); cudaFree(mc->d_tok);
    cudaFree(mc->d_dpos);
    if (mc->graph) cudaGraphExecDestroy(mc->graph);
    free(mc->l);
    free(mc);
    t->cuda = NULL;
}


#endif // MTP_CUDA_CUH
