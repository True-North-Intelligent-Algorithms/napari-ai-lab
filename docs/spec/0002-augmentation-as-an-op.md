# 0002 — Augmentation moves to scikit-ops

**Status:** proposed. Nothing built. Depends on
[scikit-ops 0015](../../../scikit-ops/docs/design/0015-augment-ops.md),
which is also unbuilt.

A narrower companion to [0001](0001-what-moves-to-scikit-ops.md). 0001 put
`Augmenters/` in the **stays** column, on the grounds that training is an
orchestrator role. That was too coarse. The orchestration stays; the transform
inside it does not.

## Why this one first

It is the only piece of the training story that is buildable now.

Every other row needs something that does not exist — the path/artifact role
for custom-model inference, session handles for the SAM-backed segmenters, a
training op of any kind. Augmentation needs none of them: arrays in, arrays out,
in an environment. It is also the smallest, and it settles the truth-role
question (how does an op know to use nearest-neighbour on a label image) that
every training op will need answered later anyway.

It does not block, and is not blocked by, the segmenter migration in 0001.

## The seam already exists

`AugmenterBase` is already split along the line we want:

| Method | Shape | Verdict |
| --- | --- | --- |
| `augment(im, mask, patch_size, axis)` | arrays in, arrays out | **This is the op.** Already the right shape |
| `augment_and_save(...)` | normalizes, calls `augment`, names files, writes two TIFFs | **Stays.** Loop, layout, manifest |
| `create_valid_coordinates(sparse_annotation, ...)` | picks legal crop starts from what is annotated | **Stays.** Needs to know which regions the user actually labelled |
| `compute_global_normalization_stats`, `normalize_image` | percentile / global-stats normalization | **Goes, separately** — `skop.ops.normalize.percentile` already exists. Compose two ops rather than folding one into the other |

So the migration is: `augment()`'s body becomes a call, and everything around it
is untouched. Same adapter tactic as 0001 step 1 — the callers in
`image_data_model.py` (`augment_and_save` at :1840 and :1969) never notice.

## What moves and what stays

| Stays here | Goes to skop |
| --- | --- |
| `AugmenterBase` as the registry and the save path | The albumentations pipeline itself |
| `create_valid_coordinates` and the bbox-driven region choice | The geometric transform, and truth consistency |
| The patch count, and which images contribute | The final random crop to `patch_size` |
| Filenames, `input_dir` / `ground_truth_dir`, `write_info` | |
| `nd_easy_augment.py` and the whole UX | |

The dividing question, as in 0001: does it need to know what a project is?
Choosing *where* to crop needs the sparse annotation, so it stays. Applying a
flip does not, so it goes.

## Three things that change, not just move

**1. The seed becomes ours to supply, per call.** `SimpleAugmenter` has a
`seed` field; `AlbumentationsAugmenter` calls `np.random.randint` in its body
and cannot be replayed at all. An op must be deterministic given
`(pair, params, seed)`, so the sequence becomes the caller's — 4000 patches
means 4000 seeds we chose. That is strictly better: a patch set becomes
reproducible, and `write_info` can record the seed and parameters that made it.

**2. Batching.** A round-trip per patch across the environment boundary is the
wrong granularity. The op takes `n` and returns `n` variants of one pair, so
`augment_and_save` grows an inner loop over a returned batch instead of calling
out 4000 times. This is the one place the calling code genuinely restructures.

**3. `SimpleAugmenter` is not redundant.** Earlier framing had it as a subset of
albumentations; it is not. It is the **n-D** random crop — `patch_size` matches
`im.ndim`, so it works on a volume natively — while albumentations is 2-D,
applied slice-by-slice with `ReplayCompose`. It is the *simple* one that handles
3-D properly. Keep it, and treat the pair as the two-implementation case that
the skop spec has to keep substitutable.

## Steps

1. **Land the op in skop.** Segmentation-shaped first: `ImageData` in,
   `LabelsData` truth, batch out. Nothing here changes yet.
2. **`AlbumentationsAugmenter.augment()` becomes a skop call.** Keep the class,
   the dataclass fields, and the registry. Body only.
3. **Thread the seed.** `augment_and_save` takes a seed; the caller in
   `image_data_model.py` owns the sequence; `write_info` records it.
4. **Batch.** `augment` returns `n` pairs; `augment_and_save` writes them.
5. **Lift normalization out**, to `skop.ops.normalize.percentile` composed
   before the augment call — or decide deliberately to keep global-stats
   normalization here, since the "global" in it is a project-level statistic and
   may genuinely not be an op's business.
6. **Then `SimpleAugmenter`**, once skop has an n-D implementation to call.
   Not before — moving it early would lose the 3-D path.

Steps 1–2 are worth doing alone. If nothing after them ever happens, the code is
no worse off than today.

## Open questions

- **The 3-D story is decided in skop, not here** (see that spec's open
  questions), but ai-lab is the caller that has 3-D data, so we are the ones who
  find out whether per-slice replay is good enough. Worth an experiment before
  step 6, not after.
- **Global normalization stats.** `use_global_stats` computes over the whole
  project. An op cannot do that. Either we normalize before calling, or the
  stats become a parameter we pass in. Leaning the latter.
- **Parameter declaration.** The `metadata={"param_type": "augmentation"}`
  dataclass fields are the same hand-rolled mechanism 0001 wants replaced by op
  signatures. Same fix, same open question about whether `nd_easy_augment`'s form
  is generated or stays hand-built.
- **Does the augmenter registry survive** once ops are discovered by
  `skop.discovery`? Same question 0001 asks of `SegmenterBase`, and it should
  get the same answer.
