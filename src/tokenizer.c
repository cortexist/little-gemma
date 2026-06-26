#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "tokenizer.h"

// ---- string -> int hash map (open addressing) -----------------------------

struct map {
    const char **keys;   // borrowed C strings (NULL = empty slot)
    int         *vals;
    size_t       cap;    // power of two
};

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h;
}

static void map_init(struct map *m, size_t count) {
    size_t cap = 1;
    while (cap < count * 2) cap <<= 1;
    m->cap = cap;
    m->keys = calloc(cap, sizeof(char *));
    m->vals = malloc(cap * sizeof(int));
}

static void map_free(struct map *m) { free((void *)m->keys); free(m->vals); }

static void map_put(struct map *m, const char *key, int val) {
    size_t mask = m->cap - 1, i = fnv1a(key, strlen(key)) & mask;
    while (m->keys[i]) {
        if (strcmp(m->keys[i], key) == 0) { m->vals[i] = val; return; }
        i = (i + 1) & mask;
    }
    m->keys[i] = key;
    m->vals[i] = val;
}

// Look up a key given as (pointer, length) rather than NUL-terminated.
static int map_get(const struct map *m, const char *s, size_t n) {
    size_t mask = m->cap - 1, i = fnv1a(s, n) & mask;
    while (m->keys[i]) {
        if (strlen(m->keys[i]) == n && memcmp(m->keys[i], s, n) == 0) return m->vals[i];
        i = (i + 1) & mask;
    }
    return -1;
}

// ---- tokenizer ------------------------------------------------------------

// A control / user-defined token, matched atomically before BPE (e.g. <start_of_turn>).
struct special_tok { const char *str; int id; int len; };

struct tokenizer {
    struct map  vocab;       // token text -> id
    struct map  merges;      // "left right" -> rank
    char      **id_to_text;  // borrowed: tokens array
    int         n_vocab;
    int         bos, eos;
    struct special_tok *special;  // sorted by length descending (longest match first)
    int         n_special;
};

static int cmp_special_desc(const void *a, const void *b) {
    return ((const struct special_tok *)b)->len - ((const struct special_tok *)a)->len;
}

struct tokenizer *tokenizer_init(const struct gguf_context *ctx) {
    const struct gguf_meta *toks = gguf_find_meta(ctx, "tokenizer.ggml.tokens");
    const struct gguf_meta *mrg  = gguf_find_meta(ctx, "tokenizer.ggml.merges");
    if (!toks || toks->type != GGUF_TYPE_ARRAY) return NULL;

    struct tokenizer *tk = calloc(1, sizeof(*tk));
    if (!tk) return NULL;
    tk->id_to_text = toks->value.arr.data;
    tk->n_vocab    = (int)toks->value.arr.n;
    tk->bos = (int)gguf_get_u32(ctx, "tokenizer.ggml.bos_token_id", 2);
    tk->eos = (int)gguf_get_u32(ctx, "tokenizer.ggml.eos_token_id", 1);

    map_init(&tk->vocab, tk->n_vocab);
    for (int i = 0; i < tk->n_vocab; i++) map_put(&tk->vocab, tk->id_to_text[i], i);

    if (mrg && mrg->type == GGUF_TYPE_ARRAY) {
        char **m = mrg->value.arr.data;
        int n = (int)mrg->value.arr.n;
        map_init(&tk->merges, n);
        for (int i = 0; i < n; i++) map_put(&tk->merges, m[i], i);  // rank = index
    } else {
        map_init(&tk->merges, 1);
    }

    // Collect control (3) and user-defined (4) tokens, matched atomically.
    const struct gguf_meta *tt = gguf_find_meta(ctx, "tokenizer.ggml.token_type");
    const int32_t *types = (tt && tt->type == GGUF_TYPE_ARRAY &&
                            tt->value.arr.type == GGUF_TYPE_INT32) ? tt->value.arr.data : NULL;
    if (types) {
        int cap = 0;
        for (int i = 0; i < tk->n_vocab; i++)
            if ((types[i] == 3 || types[i] == 4) && tk->id_to_text[i][0]) cap++;
        tk->special = malloc((size_t)cap * sizeof(*tk->special));
        for (int i = 0; i < tk->n_vocab; i++) {
            if ((types[i] == 3 || types[i] == 4) && tk->id_to_text[i][0]) {
                tk->special[tk->n_special].str = tk->id_to_text[i];
                tk->special[tk->n_special].id  = i;
                tk->special[tk->n_special].len = (int)strlen(tk->id_to_text[i]);
                tk->n_special++;
            }
        }
        qsort(tk->special, tk->n_special, sizeof(*tk->special), cmp_special_desc);
    }
    return tk;
}

void tokenizer_free(struct tokenizer *tk) {
    if (!tk) return;
    map_free(&tk->vocab);
    map_free(&tk->merges);
    free(tk->special);
    free(tk);
}

const char *tokenizer_token_text(const struct tokenizer *tk, int id) {
    return (id >= 0 && id < tk->n_vocab) ? tk->id_to_text[id] : NULL;
}

int tokenizer_token_id(const struct tokenizer *tk, const char *s) {
    return map_get(&tk->vocab, s, strlen(s));
}
int tokenizer_bos(const struct tokenizer *tk) { return tk->bos; }
int tokenizer_eos(const struct tokenizer *tk) { return tk->eos; }

int tokenizer_is_special(const struct tokenizer *tk, int id) {
    for (int s = 0; s < tk->n_special; s++) if (tk->special[s].id == id) return 1;
    return 0;
}

// ---- BPE over one newline-free segment ------------------------------------

struct sym { const char *text; int n; int prev, next; };
struct bigram { int rank, left, right, ln, rn; };

// min-heap by (rank, left)
struct heap { struct bigram *a; int n, cap; };

static int bigram_less(const struct bigram *x, const struct bigram *y) {
    return x->rank < y->rank || (x->rank == y->rank && x->left < y->left);
}
static void heap_push(struct heap *h, struct bigram b) {
    if (h->n == h->cap) { h->cap = h->cap ? h->cap * 2 : 64; h->a = realloc(h->a, h->cap * sizeof(*h->a)); }
    int i = h->n++;
    h->a[i] = b;
    while (i > 0) { int p = (i - 1) / 2; if (bigram_less(&h->a[i], &h->a[p])) { struct bigram t = h->a[i]; h->a[i] = h->a[p]; h->a[p] = t; i = p; } else break; }
}
static struct bigram heap_pop(struct heap *h) {
    struct bigram top = h->a[0];
    h->a[0] = h->a[--h->n];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = l + 1, s = i;
        if (l < h->n && bigram_less(&h->a[l], &h->a[s])) s = l;
        if (r < h->n && bigram_less(&h->a[r], &h->a[s])) s = r;
        if (s == i) break;
        struct bigram t = h->a[i]; h->a[i] = h->a[s]; h->a[s] = t; i = s;
    }
    return top;
}

static int utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;
}

static void try_bigram(const struct tokenizer *tk, struct sym *s, struct heap *h, int left, int right) {
    if (left < 0 || right < 0) return;
    char pair[1024];
    int ln = s[left].n, rn = s[right].n;
    if (ln + 1 + rn >= (int)sizeof(pair)) return;
    memcpy(pair, s[left].text, ln);
    pair[ln] = ' ';
    memcpy(pair + ln + 1, s[right].text, rn);
    int rank = map_get(&tk->merges, pair, ln + 1 + rn);
    if (rank < 0) return;
    heap_push(h, (struct bigram){ rank, left, right, ln, rn });
}

static void bpe_segment(const struct tokenizer *tk, const char *seg, int len,
                        int *out, int *n_out, int max) {
    if (len <= 0) return;
    struct sym *s = malloc((size_t)len * sizeof(*s));
    int ns = 0;
    for (int i = 0; i < len; ) {
        int cl = utf8_len((unsigned char)seg[i]);
        if (i + cl > len) cl = 1;
        s[ns].text = seg + i; s[ns].n = cl;
        s[ns].prev = ns - 1; s[ns].next = (i + cl < len) ? ns + 1 : -1;
        i += cl; ns++;
    }

    struct heap h = {0};
    for (int i = 1; i < ns; i++) try_bigram(tk, s, &h, i - 1, i);

    while (h.n > 0) {
        struct bigram b = heap_pop(&h);
        if (s[b.left].n != b.ln || s[b.right].n != b.rn) continue;  // outdated
        s[b.left].n += s[b.right].n;          // absorb right into left
        s[b.right].n = 0;
        s[b.left].next = s[b.right].next;
        if (s[b.right].next >= 0) s[s[b.right].next].prev = b.left;
        try_bigram(tk, s, &h, s[b.left].prev, b.left);
        try_bigram(tk, s, &h, b.left, s[b.left].next);
    }
    free(h.a);

    for (int i = 0; i != -1; i = s[i].next) {
        if (s[i].n == 0) continue;
        int id = map_get(&tk->vocab, s[i].text, s[i].n);
        if (id >= 0) {
            if (*n_out < max) out[(*n_out)++] = id;
        } else {                                // byte fallback: <0xXX> per byte
            for (int b = 0; b < s[i].n; b++) {
                static const char *hex = "0123456789ABCDEF";
                unsigned char ch = (unsigned char)s[i].text[b];
                char t[7] = { '<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>', 0 };
                int bid = map_get(&tk->vocab, t, 6);
                if (bid >= 0 && *n_out < max) out[(*n_out)++] = bid;
            }
        }
    }
    free(s);
}

// Encode a stretch of ordinary text (no special tokens): spaces -> ▁, then BPE
// each newline-delimited run.
static void encode_normal(const struct tokenizer *tk, const char *text, size_t len,
                          int *out, int *n_out, int max) {
    if (len == 0) return;
    char *norm = malloc(len * 3 + 1);
    size_t nl = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] == ' ') { norm[nl++] = (char)0xE2; norm[nl++] = (char)0x96; norm[nl++] = (char)0x81; }
        else norm[nl++] = text[i];
    }
    size_t i = 0;
    while (i < nl) {
        int nlrun = (norm[i] == '\n');
        size_t j = i;
        while (j < nl && (norm[j] == '\n') == nlrun) j++;
        bpe_segment(tk, norm + i, (int)(j - i), out, n_out, max);
        i = j;
    }
    free(norm);
}

// Does any special token match at text[i]? Returns its index (longest match, as
// the list is sorted by length desc) or -1.
static int match_special(const struct tokenizer *tk, const char *p, size_t n) {
    for (int s = 0; s < tk->n_special; s++) {
        int len = tk->special[s].len;
        if ((size_t)len <= n && memcmp(p, tk->special[s].str, len) == 0) return s;
    }
    return -1;
}

int tokenizer_encode(const struct tokenizer *tk, const char *text, int *out, int max) {
    int n_out = 0;
    if (n_out < max) out[n_out++] = tk->bos;

    size_t len = strlen(text), seg = 0, i = 0;
    while (i < len) {
        int s = match_special(tk, text + i, len - i);
        if (s >= 0) {
            encode_normal(tk, text + seg, i - seg, out, &n_out, max);   // text before the special
            if (n_out < max) out[n_out++] = tk->special[s].id;          // the special token itself
            i += tk->special[s].len;
            seg = i;
        } else {
            i++;
        }
    }
    encode_normal(tk, text + seg, len - seg, out, &n_out, max);
    return n_out;
}
