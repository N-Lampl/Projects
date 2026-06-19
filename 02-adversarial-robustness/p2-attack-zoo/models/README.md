# models/ (git-ignored)

Trained target weights (`smallcnn.pt`) are written here by `make train` / `make attack`.
The checkpoint stores both the state dict and a small `meta` block (data source,
input channels, num classes) so the attack script can rebuild the right model. Not committed.
