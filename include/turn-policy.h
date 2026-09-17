#ifndef TURN_POLICY_H
#define TURN_POLICY_H

#include <string.h>

// Optional policy for a conversational speaker that must yield after asking
// one question. Feed complete tokenizer pieces, including channel controls.
// A question in a hidden reasoning channel must not end the spoken answer.
struct question_turn { int in_channel, channel_closed; };

// Once an answer channel has begun, opening another thought channel is a
// synthetic continuation, not a new student turn. End BEFORE caching/emitting it.
static inline int question_turn_reopens(const struct question_turn *turn,
                                        const char *piece, int special) {
    return turn->channel_closed && special && piece && !strcmp(piece, "<|channel>");
}

static inline int question_turn_feed(struct question_turn *turn,
                                     const char *piece, int special) {
    if (!piece) return 0;
    if (special) {
        if (!strcmp(piece, "<|channel>")) turn->in_channel = 1;
        if (!strcmp(piece, "<channel|>")) {
            turn->in_channel = 0;
            turn->channel_closed = 1;
        }
        return 0;
    }
    return !turn->in_channel && strchr(piece, '?') != NULL;
}

#endif
