# Human ratings — none used here

Section 3.7 generates melody from a notated corpus and from an authored
valence/arousal mapping. No listener rated a melody, and no result in the
report rests on one.

The section does READ rated data, once, and it is worth being precise about
what that is. `train/pick_melody_anchor.py` picks the timbre the melody is
voiced on by looking up Section 3.4.2's rated seed pool — a preset a listener
liked, chosen so the melody sits inside the drone rather than on top of it.
That is a lookup over another section's measurements, not a rating study of
this one, and it happens once at build time so the runtime needs no rating data
at all.

The valence/arousal mapping itself is authored and declared, never learned:
folk songs carry no affect labels. Listening judgements did shape the pace and
phrasing constants, and the code says so where they occur, but they were
auditions during development rather than a rated study.
