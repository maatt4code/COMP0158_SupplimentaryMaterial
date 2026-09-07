# Inference — empty, and that is the finding

Nothing in Section 3.4.1 reached the runtime.

Every other model directory splits into training code and an inference module
the conductor imports. This one has no inference half, because none of the four
models trained here — the inverse CVAE, the judge proxy, the proxy ensemble, and
the closed-loop mapper — is loaded by the conductor. Verified: the conductor
loads exactly one `.pt` file, the melody transformer of Section 3.7.

That absence is the result, not an oversight. Section 4.1 is the account of a
loop that optimised successfully against its own reward model and produced audio
a listener rejects. What shipped instead is the human-grounded retrieval of
Section 3.4.2, which is why that directory has an `inference/` and this one does
not.

The checkpoints in `../weights/` are here so the experiment can be reproduced
and the failure inspected, not so it can be deployed.
