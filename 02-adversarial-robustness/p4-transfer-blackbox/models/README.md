# Models

Two **deliberately different** small classifiers, trained at runtime and saved
here (git-ignored):

- `cnn.pt` — `SmallCNN`, a 2-conv-layer net with ReLU. The **surrogate** we have
  white-box access to.
- `mlp.pt` — `SmallMLP`, a fully-connected net with Tanh. The **target** we
  attack (black-box for the query attacks; transfer for the PGD examples).

Both are tiny and train in well under a minute on CPU. `make attack` trains them
automatically if the weights are missing; `make train` trains them explicitly.
Weights are produced at runtime and are **never committed**.
