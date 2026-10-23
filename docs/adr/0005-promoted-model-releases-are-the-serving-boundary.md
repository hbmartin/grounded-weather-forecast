# Promoted model releases are the serving boundary

## Status

Accepted.

## Context

Selecting methods by globbing every scores file made evaluation evidence and serving
state indistinguishable. A synthetic backtest could overwrite a live winner, a new
dataset could consume scores produced from an old source set, and `predict --now`
could use evidence created after the requested historical issue time. Refitting a
selected method at request time also left no durable identity for the decision being
served.

The project needs to keep exploratory evidence useful without allowing it to become a
production decision accidentally. It also needs an inspectable answer to “which
evaluation justified this value?” and an exact replay path for documents that were
actually emitted.

## Decision

Every score row belongs to an **Evaluation Run** identified by its dataset fingerprint,
source set and kind, product, window, per-variable truth semantics, method set, code
version, configuration fingerprint, and creation time.

Live backtests own elections and promotion-time live-gate changes. Only completed
products may receive new approvals. Promotion creates a **Model Release** containing
the chosen method for each product × variable × lead bucket, its original evaluation
dataset and evidence context, and its available training cutoff. Pending, failed,
unrequested, and completed slices without a qualifying replacement retain compatible
incumbents.

Current-time serving reads approved decisions without scanning scores to re-elect
methods or writing releases. It permits data-only rebuilds when code, configuration,
product sources, exact feature set, and truth semantics match; retained evidence is
never relabelled as evaluated against the newer dataset. Prediction-time safety
guards remain active. Historical reconstruction still requires its exact dataset
and a release available by the requested issue time. Forecast documents identify
the dataset actually used and carry the release identity.

Immutable release content and approval time are separate. A directory-wide ledger
lock serializes incumbent reads, elections, content publication and an atomic approval
event. Events are written only for changed decisions: A → B → A reapproves A; A → A
is unchanged. New content carries `approval_required` outside its content hash and
is invisible until its event commits. Legacy releases retain implicit approval through
`promoted_at`. All readers use the same aware timestamp validation and ordering;
historical replay considers only events available by its cutoff. Managed event
sequence determines current supersession without wall-clock eligibility checks.
Historical visibility is conservative: additive `visible_at` is the maximum of
actual approval time, stored content publication time, and preceding managed
visibility floors. Historical reads require all three clocks within the cutoff.
Older events derive floors in sequence order without artifact rewrites. Malformed,
timezone-less and overflowing clocks remain invalid. Managed events precede
implicit legacy approvals, which retain publication-time eligibility and ordering.
A corrected wall clock cannot revive a lower current sequence or invalidate a
legitimate reapproval predating stored publication. Approval events keep their
actual commit clock in `approved_at`, not the earlier snapshot clock. New objects
require approvals by default; only legacy read compatibility supplies an implicit
approval for missing markers. First approval and reapproval are distinct outcomes.

Carried choices in completed products still pass the live gate, including slices
without a new winner. Pending, failed and unrequested products remain untouched.
Combined releases union compatible per-variable contexts and reject conflicting
identity. Each selected slice preserves its available training cutoff; summaries are
recomputed after replacements, without borrowing a legacy aggregate for an unknown
slice. Context-free legacy choices can justify only their identical recorded dataset,
and retain known evaluation references without invented context or cutoffs.
An additive per-slice `legacy_context_missing` marker prevents modern contexts
sharing an evaluation ID from authorizing a context-free legacy slice on a newer
dataset. Historical `"unknown"` evaluation placeholders are absent provenance.
Final selected contexts are merged once; unused historical contexts cannot poison
promotion. Conflicts fail the transaction with typed diagnostics and exit 1,
preserving scores and approvals while maintenance continues diagnostics if the
evaluations succeeded. Real evaluation failures outrank promotion errors and stop
maintenance. Typed component diagnostics preserve both in nullable
`failure_details_json` without duplicate per-product run rows.
Blocked gate measurements are separate best-effort telemetry, so changing MAE or
sample count cannot create diagnostic-only approval churn. Promotion captures
diagnostic identity once and writes telemetry after releasing the approval lock,
independently of approval publication success.

Release fields are additive. Older single-dataset ledgers remain readable using
their recorded provenance; missing evaluation context is not invented. The active
release pointer continues to describe the locally served document.

When no compatible release evidence exists, serving enters an explicit degraded mode
and uses a fit-free equal-weight blend. Synthetic evaluations remain reportable but are
never eligible for a live release. Every emitted document is archived atomically so a
previously served issue time can be replayed byte-for-byte in meaning rather than
reconstructed with future data.

## Consequences

- Score files include window and evaluation identity, so independent runs do not erase
  one another.
- Historical prediction cannot learn from future truth or future promotion evidence.
- Cold starts remain useful but cannot masquerade as fitted grounding.
- Releases add small JSON artifacts and selection metadata to maintain.
- Methods are still fitted from the compatible as-of training matrix when a new document
  is produced; exact reproducibility of an already served document comes from the served
  document archive.
- Since 2026-08-08, selection's incumbent-retention guard reads the previous release's
  per-slice method, so a release is a function of (evidence, config, code, previous
  release) — path-dependent by design. Retention is a fixed point under fixed evidence
  (a re-run with the retained release as the previous one re-derives the same
  selections and hashes to the same id) **within one release history**. Different
  prior incumbents can settle on different path-dependent fixed points under the
  same near-tie evidence. Two different serving decisions still cannot share a
  release id.

Approval snapshots also supply report and retention readers with one cutoff and
ordering. Retention-only references from active/recent-served legacy documents
protect evidence without granting serving authorization. Unverifiable references
suspend pruning, including output-only pointer read failures. When the pointer is
unavailable, current approval fallback and recent served references are protected
together; explicit empty pointers retain their existing semantics.
Churn separates immutable publication and approval clocks with
additive nullable fields and in-memory legacy normalization; approval identity,
not only release identity, distinguishes repeated transitions.

Verification uncertainty and authorization are separate. Explicit directory scans
and typed diagnostics expose unreadable events/releases, invalid clocks/identities
and listing failures. Pruning conservatively keeps all published scores when
protection is unverifiable, including corrupt pointers or served archives; only a
genuinely missing pointer enables approval fallback. Current managed future clocks
remain valid, while uncertain implicit legacy age cannot justify deletion.

Expected publication failures and stale snapshots produce structured promotion-only
failures; completed product evidence survives, and real evaluation failures retain
precedence. One snapshot caches visibility and ordering. Compatibility readers
accept additive event fields, but rollback to a pre-#79 reader needs a backport
because existing `visible_at` events cannot be made readable by an unmodified old
binary without rewriting immutable artifacts. Conservative historical floors remain
uncapped and append-only.

Daily reads use the same pure provenance decoding without importing persistence.
Backtest attribution follows each selection's evaluation dataset. Realized evidence
may select one explicitly labelled compatible older cohort when exact-release
evidence is absent, retaining its original release and served dataset identity.
This is reporting evidence, never an approval or a new serving election.
