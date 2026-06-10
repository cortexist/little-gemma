#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "gguf.h"
#include "model.h"
#include "tokenizer.h"

#define N_GEN 256  // max tokens to generate (stops early on EOS)

// Print a token's text with the ▁ marker (U+2581 = e2 96 81) rendered as a space.
static void print_piece(const char *s) {
    if (!s) return;
    for (const char *p = s; *p; p++) {
        if ((unsigned char)p[0] == 0xE2 && (unsigned char)p[1] == 0x96 && (unsigned char)p[2] == 0x81)
            { putchar(' '); p += 2; } else putchar(*p);
    }
}

// Tokenize the prompt under the Gemma chat template, generate greedily, report tok/s.
static void generate(const struct gguf_context *ctx, const char *prompt) {
    struct tokenizer *tk = tokenizer_init(ctx);
    if (!tk) { fprintf(stderr, "tokenizer init failed\n"); return; }

    // Wrap the user prompt in gemma4's chat format so instruction-tuned models
    // respond properly. <|turn> and <turn|> tokenize as atomic special tokens;
    // BOS is added by tokenizer_encode.
    size_t cap = strlen(prompt) + 64;
    char *chat = malloc(cap);
    snprintf(chat, cap, "<|turn>user\n%s<turn|>\n<|turn>model\n", prompt);

    int promptv[4096];
    int n_prompt = tokenizer_encode(tk, chat, promptv, 4096);
    free(chat);

    struct model m;
    if (model_init(&m, ctx) != 0) { tokenizer_free(tk); return; }
    struct kvcache kv;
    if (kvcache_init(&kv, &m, n_prompt + N_GEN + 1) != 0) {
        model_free(&m); tokenizer_free(tk); return;
    }
    int pos = 0;
    // Stop at end-of-turn. gemma4's turn end is <turn|>; the gguf eos_token_id is
    // <turn|> on E2B but <eos> on 12B, so stop on either.
    int eot = tokenizer_token_id(tk, "<turn|>");
    int eos = tokenizer_eos(tk);
    // Thinking models name a channel between <|channel> and <channel|> (e.g.
    // "thought"); hide that channel name so only the content prints.
    int ch_open = tokenizer_token_id(tk, "<|channel>");
    int ch_close = tokenizer_token_id(tk, "<channel|>");
    int in_thought = 0;

    // echo the user's prompt (not the chat scaffolding)
    printf("%s\n", prompt);
    fflush(stdout);

    // prefill: the last prompt token's forward also picks the first generated token
    clock_t tp = clock();
    for (int i = 0; i + 1 < n_prompt; i++) model_forward_next(&m, &kv, promptv[i], pos++);
    int best = model_forward_next(&m, &kv, promptv[n_prompt - 1], pos++);
    double t_prompt = (double)(clock() - tp) / CLOCKS_PER_SEC;

    // greedy generation
    clock_t tg = clock();
    int g = 0;
    for (; g < N_GEN; g++) {
        if (best == eot || best == eos) break;               // end of turn
        if (best == ch_open)  in_thought = 1;                // channel name starts
        else if (best == ch_close) in_thought = 0;           // channel name ends
        else if (!in_thought && !tokenizer_is_special(tk, best)) {
            print_piece(tokenizer_token_text(tk, best));
            fflush(stdout);
        }
        best = model_forward_next(&m, &kv, best, pos++);
    }
    double t_gen = (double)(clock() - tg) / CLOCKS_PER_SEC;

    printf("\n\n");
    fprintf(stderr, "prompt: %d tokens in %.2fs (%.2f tok/s)\n",
            n_prompt, t_prompt, n_prompt / (t_prompt > 0 ? t_prompt : 1e-9));
    fprintf(stderr, "gen:    %d tokens in %.2fs (%.2f tok/s)\n",
            g, t_gen, g / (t_gen > 0 ? t_gen : 1e-9));

    kvcache_free(&kv);
    model_free(&m);
    tokenizer_free(tk);
}

int main(int argc, char **argv) {
    const char *model = NULL, *prompt = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc)      model  = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
    }
    if (!model) {
        printf("Usage: %s -m <model.gguf> [-p \"prompt\"]\n", argv[0]);
        return 1;
    }

    struct gguf_context *ctx = load_gguf(model);
    if (!ctx) return 1;

    gguf_dump(ctx);

    struct config cfg;
    if (config_load(&cfg, ctx) == 0) { printf("\n"); config_print(&cfg); }

    if (prompt) { printf("\n"); generate(ctx, prompt); }

    free_gguf(ctx);
    return 0;
}
