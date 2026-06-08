#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "gguf.h"

// A BPE tokenizer matching gemma4's scheme: spaces become U+2581 (the ▁ marker),
// text is split only on newlines, then byte-pair merges run by rank; unknown
// bytes fall back to <0xXX> tokens. The vocab/merges are borrowed from `ctx`,
// so the tokenizer must be freed before the gguf context.
struct tokenizer;

struct tokenizer *tokenizer_init(const struct gguf_context *ctx);
void              tokenizer_free(struct tokenizer *tk);

// Encode `text` into token ids (prepends BOS). Writes up to `max` ids and
// returns the count.
int tokenizer_encode(const struct tokenizer *tk, const char *text, int *out, int max);

// Token id -> its raw string (with ▁ for spaces), or NULL if out of range.
const char *tokenizer_token_text(const struct tokenizer *tk, int id);

// id of an exact token string, or -1 if not in the vocab.
int tokenizer_token_id(const struct tokenizer *tk, const char *s);

int tokenizer_bos(const struct tokenizer *tk);
int tokenizer_eos(const struct tokenizer *tk);

// Is this id a control/user-defined token (e.g. <end_of_turn>)? Used to stop
// generation and to avoid printing scaffolding tokens.
int tokenizer_is_special(const struct tokenizer *tk, int id);

#endif // TOKENIZER_H
