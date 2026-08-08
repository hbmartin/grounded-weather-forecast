# Concepts

The ideas this project is built on, in plain language. No equations, no
meteorology assumed.

Read this if [Theory and concepts](theory.md) looked intimidating, or if a
leaderboard is showing you numbers and you are not sure what they claim. When a
term is unfamiliar, the [Glossary](glossary.md) has a one-line version.

---

## 1. A forecast is not a number

"It will be 22 °C" is not a forecast. This is:

> **Open-Meteo's ECMWF model**, in the run it published **at 06:00 today**, says
> the temperature **at 15:00 today** will be 22 °C — where "at 15:00" means the
> *average over the 15:00–16:00 hour*.

Four things had to be attached before the number meant anything:

- **Who said it** — and which *model*, not just which website. Several popular
  weather apps are re-selling the same underlying model, so counting them as
  separate opinions overstates how much you know.
- **When they said it** — the *issue time*. A forecast made at 06:00 knows
  nothing about what happened at 07:00. This is the boundary that makes
  evaluation honest: to judge a forecast fairly you must judge it on what it knew.
- **What moment it is about** — the *valid time*.
- **What it is measuring** — the temperature *at* 15:00, or the *average across*
  the 15:00 hour? Providers rarely say. On a clear day the temperature climbs
  about 2 °C per hour, so guessing wrong invents about 1 °C of error that looks
  exactly like the provider being biased.

The gap between issue time and valid time is the **lead time**, and it is the
single most important thing about a forecast. A provider that is excellent three
hours out may be mediocre seven days out. So everything in this system is
measured, fitted, and decided *separately for each range of lead times*.

---

## 2. Why a personal weather station changes things

Providers forecast a **grid cell** — a square kilometres across, at some nominal
elevation. Your thermometer is in a specific yard, at a specific spot on a
hillside, possibly in the shade of a specific tree.

The difference between the two is real, systematic, and **invisible to every
provider**. Your yard might run 1.5 °C cooler than the grid cell every clear
night. No provider can know that. You can measure it.

That measurement is the whole idea. The system has one input nobody else has, and
it uses it three ways:

| Stage | What it does | Plain version |
|---|---|---|
| **Grounding** | corrects each provider toward your station | "this provider always runs 1.5 °C warm here — subtract it" |
| **Blending** | combines the corrected providers | "now average their opinions, cleverly" |
| **Anchoring** | folds in the live reading | "the yard is 2 °C warmer than everyone predicted right now, so it probably still will be in ten minutes" |

The surprising result — and it took a bug to learn it — is that **the order
matters and the first stage matters most**. Correcting is worth more than
combining. The next section explains why.

---

## 3. Why averaging more forecasts stops helping

The intuition behind averaging is that errors cancel. Average four independent
opinions and you get roughly half the error of one.

The catch is *independent*. Most consumer weather APIs repackage the same handful
of global weather models. When those models are wrong, they are wrong **together**.

An analogy: asking eight people for directions sounds better than asking one —
unless six of them read the same wrong map. Averaging their answers does not find
the right street. It just gives you a very confident average of the same mistake.

Measured on this station's data, **eight providers behaved like fewer than two
independent ones.** Their errors moved together about half the time.

Two consequences follow, and they are the reason the system is shaped the way it
is:

1. **No amount of clever weighting can remove an error that every source shares.**
   Only correcting against something *outside* the group can — which is what your
   thermometer is.
2. **Fancy weighting is worth surprisingly little.** A plain arithmetic average
   of corrected providers finishes within a hair of the most sophisticated method
   in the system. This is a known result in forecasting called the *combination
   puzzle*: estimating "optimal" weights adds more noise than it removes bias.

Which does not mean the sophisticated methods are pointless — they win specific
slices, and the leaderboard says which. It means the honest baseline is very hard
to beat, and any system claiming otherwise deserves scrutiny.

---

## 4. Bias vs. error

Two different failures, and conflating them hides the fixable one.

- **Error** is being wrong. Some of it is irreducible; weather is chaotic.
- **Bias** is being wrong *in a consistent direction*. Always 1 °C too warm.

Bias is the good kind of wrong, because it is **correctable**. Measure it,
subtract it, and it goes away.

This is why the leaderboard reports bias as its own column instead of folding it
into a single accuracy score. A method can have perfectly ordinary accuracy while
being systematically warm — and a scoreboard showing only accuracy would never
tell you.

That column earned its keep. It caught a bug where the textbook-correct way of
correcting forecasts was quietly *adding* 1.4 °C of warm bias, making forecasts
worse than doing nothing at all. Accuracy alone did not reveal it; the bias column
did. The story is [ADR 0004](adr/0004-grounding-defaults-to-bias-only.md).

---

## 5. What "probability of precipitation" claims

"30% chance of rain" does not mean 30% of your yard, or rain for 30% of the hour.
In this system it means exactly:

> Of all the hours where I said 30%, it rained in about 30% of them.

That property is called **calibration**, and it is testable — you just collect
every hour where the forecast said 30% and count. A forecaster can be perfectly
calibrated and still not very useful (always predicting the historical average is
perfectly calibrated), so calibration is necessary, not sufficient.

"It rained" needs a definition too. Here it is **0.254 mm** — one hundredth of an
inch, the standard "measurable" threshold. It is a convention, and a different
threshold would make every probability in the system mean something different.

---

## 6. What "skill" means, and why it needs a rival

Accuracy alone is unreadable. Is 1.2 °C of average error good? It depends
entirely on the climate, the season, and the lead time.

So accuracy is always reported **relative to a named rival**:

> "17% better than the best single provider" — meaning the average error is 17%
> smaller than the best individual weather API achieved on the same hours.

The rival matters enormously, which is why three are always reported:

| Rival | What beating it proves |
|---|---|
| **The best single provider** | you beat the best app you could have just used instead |
| **A plain average of the raw providers** | your *corrections* are doing something |
| **A strong corrected baseline** | your *method* is doing something, beyond the corrections |

"We beat the worst provider" is not a claim worth making. Neither is "we beat
the average" if the average was never the thing to beat.

---

## 7. Why measuring yourself is hard

The seductive failure mode in forecasting is testing on data your model already
saw. It produces beautiful numbers and useless forecasts.

The obvious guard — "only train on the past" — is not enough, and the reason is
worth internalizing:

> A forecast **made** yesterday **about tomorrow** has an issue time in the past.
> But its truth has not happened yet. Training on it means training on the future.

So the system tracks, for every forecast, not when it was made but **when its
answer became knowable** — and only trains on forecasts whose answers had already
arrived. Everything else is off limits.

Then it assumes it got that wrong anyway. There is a test that corrupts every
truth value that had not yet happened at each point in time, re-runs the entire
system, and checks the forecasts come out **bit-for-bit identical**. If any
future information reaches a model by any route, the numbers change and the test
fails. It does not require anyone to have anticipated the leak — which is the
only kind of guard worth having.

---

## 8. Why the system refuses to be impressed by itself

A weather system that grades its own homework needs discipline, because a
leaderboard read every morning will eventually show *something* looking
significant by luck alone.

Three specific traps, and roughly how each is handled:

- **Too many comparisons.** Test forty methods against three rivals across ten
  lead ranges and hundreds of results look "significant" purely by chance. The
  system controls for how many questions it asked.
- **Looking every day.** A statistical test designed to be read once is not valid
  read nightly. So the system uses tests that stay valid no matter how often you
  check — you may look whenever you like without breaking the guarantee.
- **Picking the winner.** The best-looking method is partly best because it got
  lucky, so its reported score is optimistic. The system estimates how much of
  the win was luck and reports a corrected number.

And when the evidence does not clear the bar, **it serves a known baseline rather
than an unproven winner**, and marks the forecast `degraded` so you can see it did.

The full treatment is [Methods: model selection](methods/model-selection.md).

---

## Where to go next

<div class="grid cards" markdown>

- **[Getting started](getting-started.md)** — install and run it.
- **[Glossary](glossary.md)** — every term, one line each.
- **[Theory and concepts](theory.md)** — the same ideas with the reasoning and the
  measured numbers.
- **[Methods](methods/index.md)** — the same ideas with the mathematics.
- **[Limitations](limitations.md)** — what this cannot do. **Read before trusting
  any number.**

</div>
