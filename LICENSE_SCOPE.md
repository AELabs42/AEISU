# License scope

Copyright (c) 2026 Charles Dale B. Gleason.

## MIT: code and released weights

The MIT license in LICENSE applies to the author's code, software documentation, research records and released AEISU model weights,
except the manuscript materials and third-party materials described below.
This includes the portable implementation, the author's optimized runtime and desktop demo sources, and these weight paths:

- weights/aeisu_area.pt
- weights/aeisu_bicubic.pt
- weights/aeisu_div2k_l1.pt
- optimized/native/weights/adaptive_fp8.bin
- optimized/shader/weights/aeisu_int8_fp16.pt
- desktop_demo/native/models/aeisu_mixed/adaptive_fp8.bin (the same mixed artifact as optimized/native/weights/adaptive_fp8.bin)

Retain the MIT copyright and license notice when redistributing the covered materials.
CITATION.cff supplies the preferred academic citation; citation requests do not add conditions to the MIT license.

## CC BY 4.0: manuscript materials

The author's paper and supplement text, equations, architecture illustrations, plots and original figure composition are licensed
under Creative Commons Attribution 4.0 International. This covers manuscript materials in paper/,
except the third-party image content below. See paper/LICENSE.md for the license notice and canonical terms.

## Excluded third-party content

Rendered game imagery, screenshots, image crops, and reconstructions of third-party game or benchmark imagery are not licensed
under either grant above. Their underlying copyright and other rights remain with the applicable rights holders.
This exclusion applies to image content within comparison panels as well as any retained images; original labels, plots and
figure arrangement remain covered to the extent the author has rights in them.

Third-party dependencies, source contributions, comparator models and other materials remain under their original terms.
Any applicable upstream copyright/license notice takes precedence for that material; the repository's MIT license does not replace it.
See THIRD_PARTY.md.

The private training dataset, original recordings, evaluation image corpus, full output-image archive and full AESS application
are not bundled. These licenses do not require their release and do not grant rights in materials outside this repository.
