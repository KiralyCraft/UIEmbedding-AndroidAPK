from __future__ import annotations

import logging
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

LOGGER = logging.getLogger(__name__)
_BOUNDS_PATTERN = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")

ROLE_BACKGROUND = 0
ROLE_TEXT = 1
ROLE_IMAGE = 2
ROLE_BUTTON = 3
ROLE_EDIT = 4
ROLE_LIST = 5
ROLE_SCROLL = 6
ROLE_SELECTION = 7
ROLE_NAVIGATION = 8
ROLE_WEB = 9
ROLE_CONTAINER = 10
ROLE_COUNT = 11

ATTRIBUTE_NAMES = (
    "clickable",
    "scrollable",
    "editable",
    "selected",
    "checked",
    "checkable",
    "enabled",
    "focusable",
)


@dataclass(frozen=True, slots=True)
class UINode:
    left: float
    top: float
    right: float
    bottom: float
    role: int
    attributes: tuple[float, ...]
    depth: int

    @property
    def area(self) -> float:
        return max(0.0, self.right - self.left) * max(0.0, self.bottom - self.top)


@dataclass(frozen=True, slots=True)
class ParsedUI:
    nodes: tuple[UINode, ...]
    quality: float
    raw_node_count: int


@dataclass(frozen=True, slots=True)
class ResizeGeometry:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    scaled_width: int
    scaled_height: int
    offset_x: int
    offset_y: int

    def normalized_source_to_target(self, x: float, y: float) -> tuple[float, float]:
        source_x = x * self.source_width
        source_y = y * self.source_height
        target_x = (source_x * self.scaled_width / max(self.source_width, 1)) + self.offset_x
        target_y = (source_y * self.scaled_height / max(self.source_height, 1)) + self.offset_y
        return target_x / self.target_width, target_y / self.target_height


class XMLTargetParser:
    def __init__(self, max_nodes: int = 256, min_box_area_fraction: float = 1.0e-5) -> None:
        self.max_nodes = max_nodes
        self.min_box_area_fraction = min_box_area_fraction

    def parse(self, path: str, image_width: int, image_height: int) -> ParsedUI:
        if path == "" or image_width <= 0 or image_height <= 0:
            return ParsedUI(nodes=(), quality=0.0, raw_node_count=0)
        xml_path = Path(path)
        if not xml_path.is_file():
            return ParsedUI(nodes=(), quality=0.0, raw_node_count=0)
        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError) as exc:
            LOGGER.debug("Could not parse XML %s: %s", xml_path, exc)
            return ParsedUI(nodes=(), quality=0.0, raw_node_count=0)

        nodes: list[UINode] = []
        raw_count = 0
        valid_count = 0
        for element, depth in walk_tree(root):
            raw_count += 1
            bounds = parse_bounds(element.attrib.get("bounds", ""))
            if bounds is None:
                continue
            left, top, right, bottom = bounds
            left = min(max(left, 0), image_width)
            right = min(max(right, 0), image_width)
            top = min(max(top, 0), image_height)
            bottom = min(max(bottom, 0), image_height)
            if right <= left or bottom <= top:
                continue
            normalized = UINode(
                left=left / image_width,
                top=top / image_height,
                right=right / image_width,
                bottom=bottom / image_height,
                role=coarse_role(element.attrib),
                attributes=attribute_vector(element.attrib),
                depth=depth,
            )
            if normalized.area < self.min_box_area_fraction:
                continue
            valid_count += 1
            nodes.append(normalized)

        nodes.sort(key=lambda node: (node.area, -node.depth))
        nodes = nodes[: self.max_nodes]
        quality = 0.0 if raw_count == 0 else min(1.0, valid_count / max(raw_count * 0.40, 1.0))
        return ParsedUI(nodes=tuple(nodes), quality=quality, raw_node_count=raw_count)


def walk_tree(root: ET.Element) -> Iterable[tuple[ET.Element, int]]:
    stack: list[tuple[ET.Element, int]] = [(root, 0)]
    while len(stack) > 0:
        element, depth = stack.pop()
        yield element, depth
        children = list(element)
        for child in reversed(children):
            stack.append((child, depth + 1))


def parse_bounds(value: str) -> tuple[int, int, int, int] | None:
    match = _BOUNDS_PATTERN.fullmatch(value.strip())
    if match is None:
        return None
    return tuple(int(match.group(index)) for index in range(1, 5))


def as_bool(attributes: dict[str, str], key: str, default: bool = False) -> bool:
    value = attributes.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "yes"}


def attribute_vector(attributes: dict[str, str]) -> tuple[float, ...]:
    class_name = attributes.get("class", "").lower()
    editable = "edittext" in class_name or as_bool(attributes, "editable")
    values = (
        as_bool(attributes, "clickable"),
        as_bool(attributes, "scrollable"),
        editable,
        as_bool(attributes, "selected"),
        as_bool(attributes, "checked"),
        as_bool(attributes, "checkable"),
        as_bool(attributes, "enabled", default=True),
        as_bool(attributes, "focusable"),
    )
    return tuple(float(value) for value in values)


def coarse_role(attributes: dict[str, str]) -> int:
    class_name = attributes.get("class", "").lower()
    resource_id = attributes.get("resource-id", "").lower()
    content = f"{class_name} {resource_id}"
    if any(token in content for token in ("toolbar", "navigation", "tablayout", "bottomnav", "actionbar")):
        return ROLE_NAVIGATION
    if "webview" in content:
        return ROLE_WEB
    if any(token in content for token in ("edittext", "textinput", "searchview")):
        return ROLE_EDIT
    if any(token in content for token in ("checkbox", "radiobutton", "switch", "togglebutton")):
        return ROLE_SELECTION
    if any(token in content for token in ("recyclerview", "listview", "gridview")):
        return ROLE_LIST
    if any(token in content for token in ("scrollview", "viewpager")):
        return ROLE_SCROLL
    if any(token in content for token in ("imagebutton", "button", "chip")):
        return ROLE_BUTTON
    if any(token in content for token in ("imageview", "image")):
        return ROLE_IMAGE
    if any(token in content for token in ("textview", "text")):
        return ROLE_TEXT
    return ROLE_CONTAINER


class XMLDenseTargetBuilder:
    def __init__(self, patch_size: int, attribute_count: int, occupancy_grid: int, count_bins: int) -> None:
        self.patch_size = patch_size
        self.attribute_count = attribute_count
        self.occupancy_grid = occupancy_grid
        self.count_bins = count_bins

    def build(
        self,
        parsed: ParsedUI,
        geometry: ResizeGeometry,
    ) -> dict[str, torch.Tensor]:
        grid_h = geometry.target_height // self.patch_size
        grid_w = geometry.target_width // self.patch_size
        roles = torch.zeros((grid_h, grid_w), dtype=torch.long)
        attributes = torch.zeros((grid_h, grid_w, self.attribute_count), dtype=torch.float32)
        valid = torch.zeros((grid_h, grid_w), dtype=torch.bool)
        best_area = torch.full((grid_h, grid_w), float("inf"), dtype=torch.float32)

        for node in parsed.nodes:
            left, top = geometry.normalized_source_to_target(node.left, node.top)
            right, bottom = geometry.normalized_source_to_target(node.right, node.bottom)
            x0 = max(0, min(grid_w - 1, int(math.floor(left * grid_w))))
            x1 = max(0, min(grid_w - 1, int(math.ceil(right * grid_w) - 1)))
            y0 = max(0, min(grid_h - 1, int(math.floor(top * grid_h))))
            y1 = max(0, min(grid_h - 1, int(math.ceil(bottom * grid_h) - 1)))
            target_area = max((right - left) * (bottom - top), 1.0e-8)
            for row in range(y0, y1 + 1):
                center_y = (row + 0.5) / grid_h
                if center_y < top or center_y > bottom:
                    continue
                for column in range(x0, x1 + 1):
                    center_x = (column + 0.5) / grid_w
                    if center_x < left or center_x > right:
                        continue
                    if target_area <= float(best_area[row, column]):
                        best_area[row, column] = target_area
                        roles[row, column] = node.role
                        attributes[row, column] = torch.tensor(node.attributes[: self.attribute_count])
                        valid[row, column] = True

        occupancy = torch.zeros((self.occupancy_grid, self.occupancy_grid), dtype=torch.float32)
        for node in parsed.nodes:
            center_x = (node.left + node.right) * 0.5
            center_y = (node.top + node.bottom) * 0.5
            column = min(self.occupancy_grid - 1, int(center_x * self.occupancy_grid))
            row = min(self.occupancy_grid - 1, int(center_y * self.occupancy_grid))
            occupancy[row, column] = 1.0

        node_count = len(parsed.nodes)
        count_bin = min(self.count_bins - 1, int(math.log2(max(node_count, 1))))
        return {
            "xml_roles": roles.flatten(),
            "xml_attributes": attributes.reshape(-1, self.attribute_count),
            "xml_valid": valid.flatten(),
            "xml_occupancy": occupancy.flatten(),
            "xml_count_bin": torch.tensor(count_bin, dtype=torch.long),
            "xml_quality": torch.tensor(parsed.quality, dtype=torch.float32),
        }
