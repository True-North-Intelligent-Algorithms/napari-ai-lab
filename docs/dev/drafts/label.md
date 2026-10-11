# Label

## Annotations → Labels → Patches

- **Annotations:** what you draw, on the full image. Can be unfinished.
  - **napari brush/fill:** draw on **Labels (Persistent)**, not Working.
  - **SAM:** the result appears on **Labels (Working)**. Commit it to keep it.
  - Only **Labels (Persistent)** is saved.
- **Label box:** marks "this part is done".
- **Labels:** the annotations inside a label box. Only these train.
- **Patches:** labels, augmented. The training set.

**SAVE before you augment. Augment before you train.**

> **SCREENSHOT TODO:** Label tab

## Label

- Draw labels.
- Draw a label box around what you labelled.

> **SCREENSHOT TODO:** SAM click result on Labels (Working)

> **SCREENSHOT TODO:** label box
