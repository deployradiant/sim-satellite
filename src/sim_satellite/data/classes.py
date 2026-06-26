"""Shared semantic mask class contract."""

from __future__ import annotations

OPENEARTHMAP_CLASSES = (
    "unlabeled",
    "bareland",
    "rangeland",
    "developed space",
    "road",
    "tree",
    "water",
    "agriculture land",
    "building",
)
OPENEARTHMAP_CLASS_IDS = {name: index for index, name in enumerate(OPENEARTHMAP_CLASSES)}
OPENEARTHMAP_RGB = {
    "unlabeled": (0, 0, 0),
    "bareland": (128, 0, 0),
    "rangeland": (0, 255, 36),
    "developed space": (148, 148, 148),
    "road": (255, 255, 255),
    "tree": (34, 97, 38),
    "water": (0, 69, 255),
    "agriculture land": (75, 181, 73),
    "building": (222, 31, 7),
}
