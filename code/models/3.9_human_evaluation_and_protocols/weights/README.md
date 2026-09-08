# Weights — empty, because nothing here is fitted

Section 3.9 fits no model. Its scripts compute descriptive statistics,
correlations and paired tests directly from the shipped ratings, so there is
no parameter to freeze and nothing to record a checksum for.

The one exception looks like a fit and is not: `analyse_longtrack.py` fits a
crossed-random-effects mixed model, but it is refitted on every run in a few
seconds and reported inside
`../data/results/longtrack_analysis.json` alongside its convergence flag. It
is a summary of the ratings, not an artefact the runtime consumes.

Sections that DO ship weights record them in a `SHA256SUMS` manifest here;
this section has no manifest because it has nothing to manifest.
