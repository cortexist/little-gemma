#include <stdio.h>
#include "turn-policy.h"

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; \
} } while (0)

int main(void) {
    struct question_turn turn = {0};
    CHECK(!question_turn_feed(&turn, "A short explanation.", 0));
    CHECK(!question_turn_feed(&turn, " Which state", 0));
    CHECK(question_turn_feed(&turn, "?\n", 0));

    // Hidden reasoning can contain questions; only the answer yields the floor.
    turn = (struct question_turn){0};
    CHECK(!question_turn_feed(&turn, "<|channel>", 1));
    CHECK(!question_turn_feed(&turn, "thought\nWhat should I ask?", 0));
    CHECK(!question_turn_feed(&turn, "<channel|>", 1));
    CHECK(question_turn_reopens(&turn, "<|channel>", 1));
    CHECK(!question_turn_reopens(&turn, "<|channel>", 0));
    CHECK(!question_turn_feed(&turn, "What do you think", 0));
    CHECK(question_turn_feed(&turn, "?\"", 0));

    // A fresh turn needs neither a question nor a channel (e.g. a farewell).
    turn = (struct question_turn){0};
    CHECK(!question_turn_reopens(&turn, "<|channel>", 1));
    CHECK(!question_turn_feed(&turn, "Goodbye.", 0));
    CHECK(!question_turn_feed(&turn, "<turn|>", 1));
    CHECK(!question_turn_feed(&turn, NULL, 0));
    CHECK(!question_turn_feed(&turn, "<reserved?>", 1));
    // -think 0 seeds a closed channel in the prompt, outside the output stream.
    turn = (struct question_turn){0, 1};
    CHECK(question_turn_reopens(&turn, "<|channel>", 1));
    CHECK(question_turn_feed(&turn, "Ready?", 0));
    puts("turn policy: OK");
    return 0;
}
