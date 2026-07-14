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

Serving considers only live evaluation runs that match the current dataset and were
available by the requested issue time. Promotion creates a **Model Release** containing
the chosen method for each product × variable × lead bucket, its evidence IDs, and its
training cutoff. Forecast documents and history rows carry the release identity.

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
