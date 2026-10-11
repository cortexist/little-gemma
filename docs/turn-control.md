# Optional client-controlled assistant prefix

`LG_SNAPSHOT_HIDDEN=1` exports an immutable `LGH1` hidden-state frame after
prefill. `LG_TURN_CONTROL=1` additionally requires a client action before the
assistant answer starts. Both default off; existing socket clients are unchanged.
Positive reasoning budgets are incompatible with this non-reasoning mode.

For each snapshot, write seven bytes: `0x03`, its little-endian uint32 turn
number, action byte (1=speak, 2=defer, 3=wait), and flags byte (bit 0=requested
speaker-name metadata, other bits zero). Turn numbers are scoped to a connection.
Missing, malformed or stale commands close the connection without granting
speech; reads have a five-second total bound. Snapshot bytes remain at the
original unseeded prefill boundary before any forced assistant token.

The engine can prepare an empty answer channel while the client computes its
choice. It then appends `[speak]` or `[defer]` to the same assistant KV turn and
continues MTP decoding. Required name output starts with `<speaker_name>`.
Silent defer/wait closes immediately when no name is requested; otherwise it
ends after the name's closing tag. No second user prompt, inference request,
unheard candidate answer or separate semantic decision is inserted in history.
The client owns wait semantics, floor/audio gating and identity validation.

Forced prefixes are not independent text-head decisions. They are listed as
`forced_tokens` separately from generated tokens. `turn_control` logs residual
command wait, channel preparation and forced-prefix processing in milliseconds;
existing prefill timing alone does not include these costs. CPU policy and
channel preparation may overlap, but answer generation depends on the action.
Clients must validate command/response association and suppress audio on error.
The engine contains no classifier weights or probability thresholds.
