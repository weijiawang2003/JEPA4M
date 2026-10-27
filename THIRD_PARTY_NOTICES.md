# Third-party references and notices

This export contains local research implementations and reports. Upstream source
trees, downloaded papers, third-party checkpoints and external workload caches
are excluded. The notices below identify the versions used or inspected; they
do not grant a repository-wide license for this project. No such project license
was supplied in the source workspace.

## Reference implementations and methods

| Reference | Pinned revision and observed license | Use in this project |
|---|---|---|
| [CayleyPy Cube](https://github.com/khoruzhii/cayleypy-cube) | `f02604fa7b665b82e5fbe5692b336a4fe4a01bdc`; [MIT](https://github.com/khoruzhii/cayleypy-cube/blob/f02604fa7b665b82e5fbe5692b336a4fe4a01bdc/LICENSE) | Official Cube reference. The local common-search adapter derives a facelet/color mapping and imports the separately obtained upstream `Pilgrim` implementation at runtime. Neither that implementation nor its weights are bundled. |
| [LeJEPA](https://github.com/galilai-group/lejepa) | `c293d291ca87cd4fddee9d3fffe4e914c7272052`; CC BY-NC 4.0, as recorded in the pinned-source audits | Source of the SIGReg mathematical specification. The local `models.py:sigreg` expresses the single-device, two-dimensional-input statistic; local callers draw projection axes separately for the two state branches. Attribution: Balestriero and LeCun, [LeJEPA](https://arxiv.org/abs/2511.08544). The upstream license includes a noncommercial restriction; it is not an MIT reference. |
| [LeWorldModel](https://github.com/lucas-maes/le-wm) | `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`; MIT, as recorded in the pinned-source audits | Reference for action-conditioned latent prediction with an additive SIGReg objective and gradients through both encoder branches. Attribution: Lucas Maes et al., [LeWorldModel](https://arxiv.org/abs/2603.19312). The training package is not bundled. |
| [Halide](https://github.com/halide/Halide) | `d6a2683d10535b4fd3f4f3c6626255ed204dfe15`; selected core/test material covered by the top-level MIT-style notice | The workload-preparation script retrieves `test/correctness/simplify.cpp`, `src/IR.h` and the complete license, verifies source hashes, and parses a restricted integer-polynomial fragment. The upstream C++ files and raw derived panel are not bundled. Halide's composite license also contains notices for other directories; this is not a claim that all Halide components use MIT. |

The [SIGReg audit](docs/JEPA_OBJECTIVE_AUDIT.md) and
Phase 3 reference check (historical artifact omitted from this compact release) record the inspected
files and license versions. The latter explicitly records that no upstream
implementation was copied into the independent `geometry_reference.py`
diagnostic helper; that finding is scoped to the helper. The local SIGReg
adaptation is documented in those audits, rather than represented as an original
method. See the [official adapter record](docs/OFFICIAL_ADAPTER.md) and
[Cube reference audit](docs/REFERENCE_AUDIT.md) for the external model lineage.

Upstream terms apply to any separately obtained upstream material. The LeJEPA
terms are [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/);
the pinned LeWorldModel [license file](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/LICENSE)
remains the authority for its copyright notice. Other cited papers and software
are research references, not bundled dependencies or endorsements.

## CayleyPy Cube notice

The following is preserved verbatim from the pinned reference's license and
applies to that upstream work, not to this project as a whole.

```text
MIT License

Copyright (c) 2024 Khoruzhii Kirill

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Halide selected-source notice

The following is preserved verbatim from the top section of the pinned
[composite license](https://github.com/halide/Halide/blob/d6a2683d10535b4fd3f4f3c6626255ed204dfe15/LICENSE.txt),
covering the selected core/test source. The remaining notices apply to other
upstream components, none of which are bundled here.

```text
Copyright (c) 2012-2020 MIT CSAIL, Google, Facebook, Adobe, NVIDIA CORPORATION, and other contributors.

Developed by:

  The Halide team
  http://halide-lang.org

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
