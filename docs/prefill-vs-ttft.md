# Prefill rate vs TTFT — related, not 1:1

Prefill rate is a **throughput**: tokens per second through prompt
processing, a slope. TTFT (time to first token) is a **latency**: the clock
from the last input byte to the first output token, the number a user
feels. For one specific kind of turn — text-only, arriving all at once, on
a warm server, long enough to amortize overheads — the two collapse into
each other, and the README's table shows it: the A5000 12B text row
prefills 928 tokens at 1,803 tok/s, and 928 / 1803 = 0.515 s ≈ the measured
0.51 s TTFT. Everywhere else, dividing tokens by a rate underestimates,
overestimates, or simply measures a different thing. The intercepts:

**The first decode step.** A prefill rate ends when the prompt's kv is
written; TTFT ends one step later, after a full decode pass picks the first
token. That pass is decode-priced (≈16 ms on the A5000 12B, ≈115 ms on the
Orin), and the prefill→decode transition adds its own fixed cost (~66 ms
measured on the A5000, both stacks — real GPU work, not API overhead).
Short turns are dominated by terms like these.

**Chunk padding.** Prefill runs in padded launches (columns rounded to
%32/%64 for the fat single-launch kernels), so a 157-token camera turn
prefills 192 columns. The headline rate was measured on ~930-token turns
where the padding vanishes into the total; on a short turn it does not.

**The rate is width-dependent.** 3,717 tok/s is the E4B's 929-token figure;
the same machine runs a 150-token turn at ~1,300 tok/s — launches underfill
and the fixed costs amortize over less. Dividing a short turn by the
long-turn rate flatters every stack.

**Media encoders.** Encoding is always inside TTFT, but only *sometimes*
inside a reported prefill number — llama-server's `prompt_ms` excludes its
media encoder (visible in our A/B as a 0.25–0.43 s gap between its TTFT and
its prompt time on Orin image turns), while little-gemma's serve stat books
everything after the burst. Before dividing or comparing, check what a
number includes.

**Arrival overlap — where the ratio breaks completely.** TTFT starts at the
*last* input byte. When input arrives over time — video frames over a
socket, a transcript streaming from whisper — prefill can hide under the
arrival: an 849-token video turn on the Orin 12B would cost 4.9 s by
tokens÷rate, and answers in 2.3 s, because six of its seven spans were
already seated when the question landed. No rate measurement can see this;
it is a property of the serving architecture (media prefills as it arrives,
one span held back; see the README's multimodal section). The reverse also
holds: a stack can post a higher prefill rate and still lose TTFT — the
README table's media rows do exactly that.

**Warm state.** First turns pay one-time costs (weight upload or page-in,
lazy init, allocator warm-up — llama-server's first Orin turn runs 2–7×
slow), and prompt-prefix caching can make a repeated benchmark prompt skip
prefill entirely. Every number in the README table is a warm turn with the
first discarded and caching off.

A serviceable rule of thumb, for a burst-arrival turn:

    ttft ≈ encoder (if not hidden)
         + padded_columns / rate_at_that_width
         + first decode step (+ transition)

and for paced arrival, replace the middle term with whatever prefill work
remains unseated when the last byte lands — which good streaming drives
toward zero.

Measurement discipline for cross-stack TTFT (this is how the README table
was made, harness in `.scratch/ttft_llama.py`): the same client-side
stopwatch on both stacks (last request byte → first streamed token), same
GGUFs, warm servers, first turn discarded, prompt caching explicitly off,
pinned clocks, and the input-token counts printed next to the times so
policy asymmetries (e.g. a fork upscaling every image to a 252-token
minimum) are visible instead of laundered into the verdict.
