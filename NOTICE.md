# Licensing and Artifact Notice

This repository contains code only. It does not redistribute third-party
satellite datasets, extracted dataset tiles, prepared tensor data, trained
checkpoints, or model weights.

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

If these artifacts are published separately, they should be released as
non-commercial research artifacts under CC BY-NC-SA 4.0, with attribution to
the underlying datasets and with notes describing any label remapping,
filtering, or mixing steps.

Commercial use of dataset derivatives, trained checkpoints, model weights, or
other artifacts derived from non-commercial source datasets is not permitted by
this repository's license policy.

## Public Release Process

Before any public research release, create a clean copy of the repository for
publication. The release copy should include only code, configs, synthetic test
fixtures, documentation, and reproduction instructions. It should exclude local
data caches, prepared tensors, logs, checkpoints, model weights, generated
images, and any other dataset-derived artifacts.
