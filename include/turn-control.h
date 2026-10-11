#ifndef TURN_CONTROL_H
#define TURN_CONTROL_H
#include <stdint.h>
#include <string.h>

// Client reply to the current hidden snapshot: marker, LE turn, action, flags.
// No user prompt or second generation. Unknown/stale commands fail closed.
#define TURN_CONTROL_BYTES 7
#define TURN_CONTROL_SPEAK 1
#define TURN_CONTROL_DEFER 2
#define TURN_CONTROL_WAIT 3
#define TURN_CONTROL_NAME 1
struct turn_control { unsigned action, name, matched; };
static inline int turn_control_parse(struct turn_control *out,
                                     const unsigned char *p, uint32_t turn) {
    uint32_t got = (uint32_t)p[1] | (uint32_t)p[2]<<8 | (uint32_t)p[3]<<16 | (uint32_t)p[4]<<24;
    if (p[0] != 3 || got != turn || p[5] < 1 || p[5] > 3 || (p[6] & ~1u)) return -1;
    out->action = p[5]; out->name = p[6]; out->matched = 0;
    return 0;
}
static inline int turn_control_name_done(struct turn_control *c, const char *piece) {
    if (!c->name || c->action == TURN_CONTROL_SPEAK || !piece) return 0;
    const char *end = "</speaker_name>";
    for (const char *p = piece; *p; p++) {
        if (*p == end[c->matched]) c->matched++;
        else c->matched = *p == end[0] ? 1 : 0;
        if (!end[c->matched]) return 1;
    }
    return 0;
}
#endif
