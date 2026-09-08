# Inference — empty, because this section measures rather than synthesises

Nothing in Section 3.9 runs at synthesis time.

Every other model directory splits into training code and an inference module
the conductor imports. This one is an evaluation section: it takes ratings
collected from listeners and computes what they say. There is no model to
load, no parameter to sample from, and nothing for the runtime to call.

What Section 3.9 produced instead is **evidence**, and that evidence reached
the runtime by changing decisions rather than by shipping code:

- the component-preference study set the conductor's defaults — which melody
  style, which reverb, how loud the background bed sits;
- the recency result ended the attempt to model valence from movement, because
  a retrospective rating of a multi-minute track measures its last few seconds
  and so cannot serve as a target for a whole trajectory.

Look in [`../train/`](../train) for the four analysis scripts, and in
[`../human_ratings/`](../human_ratings) for the data they read. The absence of
an inference half here is the correct shape for the section, not an omission.
