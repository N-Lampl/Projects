# models/ (git-ignored)

The detector trains in well under a second on CPU, so weights are produced fresh on every
`make detect` rather than persisted. If you add a `--save` path later, write artifacts here;
nothing in this folder is committed.
