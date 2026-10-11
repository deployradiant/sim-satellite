# Licensing and Artifact Notice

This repository contains source code, documentation, and selected historical
research checkpoints, metrics, and evaluation figures under `artifacts/`.
Raw dataset archives and prepared tensor caches are not included.

## Code

The source code in this repository is licensed under the MIT License. See
[LICENSE](LICENSE).

## Third-Party Data

Experiments may require third-party datasets such as OpenEarthMap,
LandCover.ai, or LoveDA. Those datasets are not included in this repository and
remain governed by their original licenses and terms.

Several datasets considered by this project include non-commercial and
share-alike restrictions. Treat any data downloaded through this project as
proof-of-concept research material unless every source license has been
reviewed and explicitly permits broader use.

## Generated Artifacts

Generated artifacts are not covered by the MIT code license when they are
derived from non-commercial/share-alike datasets. This includes:

- prepared manifests and dataset derivatives
- prepared tensor trees
- trained checkpoints and model weights
- evaluation outputs derived from restricted datasets

The selected research artifacts published in this repository are provided as
non-commercial research artifacts under CC BY-NC-SA 4.0, with attribution to
the underlying datasets and with notes describing any label remapping,
filtering, or mixing steps.

Commercial use of dataset derivatives, trained checkpoints, model weights, or
other artifacts derived from non-commercial source datasets is not permitted by
this repository's license policy.

## Included Figures and Checkpoints

The README links to the saved 200,000-step VAE reconstruction grid and the
200,000-step segmentation comparison. These figures contain OpenEarthMap
imagery, reference labels, reconstructions, and model predictions. Original
imagery and labels retain their source terms; the MIT license applies to code.
The research-artifact terms above do not replace upstream rights or attribution.

Dataset attribution: Xia et al., *OpenEarthMap: A Benchmark Dataset for Global
High-Resolution Land Cover Mapping*, WACV 2023.
See the [OpenEarthMap overview](https://open-earth-map.org/overview.html) and
[source imagery and licenses](https://open-earth-map.org/).
Run settings and the distinction between these saved evaluations and later
local experiments are documented in [Reproduction.md](Reproduction.md).

## Public Release Scope

The published repository includes code, configs, tests, documentation,
reproduction instructions, and the selected historical research artifacts
listed above. Local data caches, prepared tensors, logs, credentials, and
unpublished experiments remain outside the release. New generated artifacts
are ignored by default through `.gitignore`.
