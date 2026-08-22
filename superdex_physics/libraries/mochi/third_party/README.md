# Vendored third-party libraries

Mochi vendors most of its third-party dependencies rather than fetching them, and some carry
local modifications.

**Every local change is marked in-line with `[Mochi]`.** The code is the record; there is no
list to keep in step with it. To find every change in a library before upgrading it:

```bash
grep -rn '\[Mochi\]' arvr/libraries/mochi/third_party/<library>/
```

Changes are made in place rather than as a patch series, so the tree builds straight from a
checkout with no patch step. The cost is that an upgrade is a manual re-apply, and the `[Mochi]`
marker is what makes that tractable: grep the outgoing tree, drop in the new release, work
through the hits.

Two things to know when re-applying:

- **Mark both ends of a block.** A commented-out or `if (FALSE)`-disabled region needs a marker
  on the opening *and* the closing line, or a re-apply can restore one without the other and
  leave the file unbalanced.
- **Not everything here is vendored.** Some dependencies are fetched at configure time by
  `third_party/CMakeLists.txt` instead, and upgrading those is a version bump rather than a
  directory replacement. Check for a directory before assuming.

When you add a local change, add the marker. That is the whole convention.
