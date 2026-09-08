"""What the component-preference ratings say, per test.

Reads the flat Space table (one row per rating, `params` carrying the stimulus
settings as JSON) and answers, per test, the one question the conductor needs:
which setting should it default to?

Deliberately NOT `export.py`. That splits the table back into the per-study
schemas the older analysis scripts read; this reads the table as it is and
reports. Neither touches the source rows.

Scoring: negative/neutral/positive -> -1/0/+1; `response2` is the follow-up
("would you use it?") -> yes/maybe/no = 1/0.5/0. A/B tests are counted against
the flag in `params` that says which side was which, so a rater's A/B position
bias cannot look like a result.

THIS IS THE MERGE OF RECORD. `load()` below is what produced the dataset the
report analysed, and it is shipped rather than rewritten for one reason: the
rating app wrote one CSV per boot, so a session that survived a restart is
split across shards, and shards from different pulls overlap. A naive
concatenation double-counts those rows and yields a different dataset from the
one the thesis reports. Three lines carry that:

  * concatenate every shard;
  * drop duplicates on (rater, test, item_id, timestamp) -- boot shards
    overlap when the app restarts mid-session;
  * drop the `setup_check` row, which is a deployment smoke test, not a rating.

Run:
  python analyse_space_ratings.py                  # the shipped ratings
  python analyse_space_ratings.py --ratings <dir>
  python analyse_space_ratings.py --json out.json
"""

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SECTION = HERE.parent
# The pseudonymised shards ship with this section; no download is needed.
DEFAULT_IN = SECTION / "human_ratings"
SCORE = {"negative": -1, "neutral": 0, "positive": 1}
USE = {"yes": 1.0, "maybe": 0.5, "no": 0.0}
# every params key the report reads, flattened into columns
PARAM_KEYS = ("effect", "strength", "reverb", "treatment", "style", "level_db",
              "bed_type", "a_is_staged", "a_is_reverb", "pair_kind",
              "has_melody", "clip_id", "drone", "bed_id")


def load(src):
    files = sorted(glob.glob(str(Path(src) / "space_ratings_*.csv")))
    if not files:
        raise SystemExit(f"\nno rating shards under:\n    {src}\n\n"
                         f"The pseudonymised shards ship in this section's "
                         f"human_ratings/.\n")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # boot shards overlap when a Space restarts mid-session
    df = df.drop_duplicates(subset=["rater", "test", "item_id", "timestamp"])
    df = df[df.rater != "setup_check"]                  # deploy smoke row
    p = df.params.apply(lambda s: json.loads(s) if isinstance(s, str) else {})
    for k in PARAM_KEYS:
        df[k] = p.apply(lambda d: d.get(k))
    df["score"] = df.response.map(SCORE)
    df["use"] = df.response2.astype(str).str.lower().map(USE)
    return df


def rank(sub, by):
    """Mean score and use-rate per condition, best first."""
    g = sub.groupby(by, dropna=False)
    return pd.DataFrame({"n": g.size(),
                         "mean_score": g.score.mean().round(2),
                         "use_rate": g.use.mean().round(2)}
                        ).sort_values("mean_score", ascending=False)


def ab_test(sub, flag, label_true, label_false):
    """Forced-choice tally, counted against `flag` so position bias cannot pose
    as a result. Returns (wins_true, wins_false, ties, p)."""
    from scipy import stats
    r = sub[sub.response.isin(["A", "B"])]
    won = (((r.response == "A") & (r[flag] == True))        # noqa: E712
           | ((r.response == "B") & (r[flag] == False)))    # noqa: E712
    t, f = int(won.sum()), int(len(r) - won.sum())
    ties = int((sub.response == "same").sum())
    p = float(stats.binomtest(t, t + f, 0.5).pvalue) if t + f else float("nan")
    print(f"  {label_true} {t} : {f} {label_false}   ties {ties}   p={p:.3f}")
    per = sub[sub.response.isin(["A", "B"])].assign(won=won).groupby("rater").won.agg(["sum", "size"])
    print("  per rater:  " + ",  ".join(f"{i} {int(r0)}/{int(r1)}"
                                        for i, (r0, r1) in per.iterrows()))
    return t, f, ties, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=str(DEFAULT_IN))
    ap.add_argument("--json", default=None, help="write findings here")
    args = ap.parse_args()

    df = load(args.ratings)
    out = {"n_rows": int(len(df)), "n_raters": int(df.rater.nunique())}
    print(f"{len(df)} ratings | {df.rater.nunique()} raters | "
          f"{df.test.nunique()} tests\n")
    print(df.rater.value_counts().to_string(), "\n")

    for title, test, by in (
            ("MOVEMENT (modulation)", "modulation", ["effect", "strength"]),
            ("REVERB on a held drone", "reverb_clips", ["reverb"]),
            ("STEREO", "stereo", ["treatment"]),
            ("MELODY by style", "melody", ["style"]),
            ("MELODY by level", "melody", ["level_db"]),
            ("BED pairing by type", "pairing", ["bed_type"]),
            ("BED pairing by level", "pairing", ["level_db"])):
        sub = df[df.test == test]
        if sub.empty:
            continue
        print(f"=== {title} ===")
        t = rank(sub, by)
        print(t.to_string(), "\n")
        out[f"{test}__{'_'.join(by)}"] = json.loads(t.reset_index().to_json(orient="records"))

    b = df[df.test == "bed_level"]
    if not b.empty:
        print("=== BED LEVEL: dB the rater settled on ===")
        counts = b.response2.astype(str).value_counts()
        print(counts.to_string())
        print("verdict:", dict(b.response.value_counts()), "\n")
        out["bed_level_choice"] = {str(k): int(v) for k, v in counts.items()}

    print("=== TRANSITIONS: reverb on a transition? ===")
    t, f, ties, p = ab_test(df[df.test == "transitions"], "a_is_reverb",
                            "reverb", "dry")
    out["transitions_reverb"] = dict(reverb=t, dry=f, ties=ties, p=p)

    print("\n=== SENSITIVITY: staged vs simultaneous morph ===")
    t, f, ties, p = ab_test(df[df.test == "sensitivity"], "a_is_staged",
                            "staged", "simultaneous")
    out["sensitivity_staged"] = dict(staged=t, simultaneous=f, ties=ties, p=p)

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
