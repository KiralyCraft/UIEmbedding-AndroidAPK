"""Read native MobileViews graph exports without modifying the extracted data.

UTG node images are full-screen representatives; ``views/`` holds widget crops.
Actions retain visits to those representatives, not additional captured images.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from ui_foundation.config import DataConfig
from ui_foundation.data.image_metadata import ImageMetadataReader
from ui_foundation.data.records import ScreenRecord

LOGGER = logging.getLogger(__name__)
MOBILEVIEWS_READER_VERSION = "mobileviews_v3"
_UTG_ASSIGNMENT = re.compile(r"^\s*(?:var|let|const)\s+utg\s*=\s*")
_SCREEN_NAME = re.compile(r"^screen_[^/\\]+\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)
_PACKAGE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+$")


def is_full_screenshot(path: Path, root: Path) -> bool:
    """Accept only contained states/screen_* files, never UI element crops."""
    return (
        path.parent.name == "states"
        and _SCREEN_NAME.fullmatch(path.name) is not None
        and path.resolve().is_relative_to(root.resolve())
        and not path.is_symlink()
        and not path.parent.is_symlink()
    )


def parse_graph_json(text: str) -> dict:
    """Parse the UTG JSON object plus JavaScript trailing commas, without eval."""
    document = text.strip().removesuffix(";").rstrip()
    try:
        graph = json.loads(document)
    except json.JSONDecodeError:
        normalized = _without_trailing_commas(document)
        if normalized == document:
            raise
        graph = json.loads(normalized)
    if not isinstance(graph, dict):
        raise ValueError("MobileViews graph must contain an object")
    return graph


def _without_trailing_commas(document: str) -> str:
    """Remove only out-of-string commas immediately before closing containers.

    A regex substitution would corrupt titles/text such as 'literal ,}'.
    All other JSON validation remains the standard decoder's responsibility.
    """
    pieces: list[str] = []
    in_string = False
    escaped = False
    for index, character in enumerate(document):
        if in_string:
            pieces.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == ",":
            after = index + 1
            while after < len(document) and document[after] in " \r\n\t":
                after += 1
            before = index - 1
            while before >= 0 and document[before] in " \r\n\t":
                before -= 1
            if (
                after < len(document) and document[after] in "]}"
                and before >= 0 and document[before] not in "[{,:"
            ):
                continue
        pieces.append(character)
    return "".join(pieces)


def read_action_visits(path: Path) -> dict[str, tuple[tuple[int, int], ...]]:
    """Map state IDs to all (continuous segment, observation position) visits.

    A row is a transition from position i to i+1. Repeated/self transitions
    remain visits. Disconnected rows start a new segment, not a fake adjacency.
    """
    visits: dict[str, list[tuple[int, int]]] = defaultdict(list)
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except FileNotFoundError:
        LOGGER.warning("Missing MobileViews actions %s; chronology is unknown", path)
        return {}
    with handle:
        rows = csv.DictReader(handle)
        # Zero-transition exports use either an empty file or a single quoted
        # empty CSV cell. Retain their screenshots with unknown chronology.
        if not rows.fieldnames or (not any(rows.fieldnames) and next(rows, None) is None):
            return {}
        if not {"from_state", "to_state", "action"}.issubset(rows.fieldnames or []):
            raise ValueError(f"Invalid MobileViews action columns: {path}")
        previous = None
        segment = -1
        position = 0
        incomplete = 0
        for row in rows:
            source = (row.get("from_state") or "").strip()
            target = (row.get("to_state") or "").strip()
            if not source or not target:
                # Some exports end with a partial CSV row. Keep known endpoint
                # observations, but never bridge a missing transition endpoint.
                incomplete += 1
                if source and source != previous:
                    segment += 1
                    visits[source].append((segment, 0))
                previous = None
                position = 0
                if target:
                    segment += 1
                    visits[target].append((segment, 0))
                    previous = target
                continue
            if source != previous:
                segment += 1
                position = 0
                visits[source].append((segment, position))
            position += 1
            visits[target].append((segment, position))
            previous = target
    if incomplete:
        LOGGER.warning("MobileViews actions %s: %d incomplete transitions; segment boundaries retained", path, incomplete)
    return {state: tuple(positions) for state, positions in visits.items()}


def read_graph_trace(config: DataConfig, graph_path: Path) -> Iterator[ScreenRecord]:
    """Yield one original full-screen representative per native UTG node."""
    # Import lazily to keep scanner dispatch and native format handling separate.
    from ui_foundation.data.scanners import canonicalize_activity

    root = Path(config.root).resolve()
    trace = graph_path.parent
    # Each worker lists only this trace's states once, never its widget crops.
    states_dir = trace / "states"
    if states_dir.is_symlink() or not states_dir.resolve().is_relative_to(root):
        raise ValueError(f"MobileViews states directory must be contained and not symlinked: {states_dir}")
    with os.scandir(states_dir) as entries:
        names = {entry.name for entry in entries if entry.is_file(follow_symlinks=False)}
    try:
        raw = _UTG_ASSIGNMENT.sub("", graph_path.read_text(encoding="utf-8-sig"), count=1)
    except FileNotFoundError:
        raw = ""
    # Accept the export's trailing commas, never execute dataset JavaScript.
    if not raw.strip():
        LOGGER.warning("Missing/empty MobileViews graph %s; reading original screen/state pairs", graph_path)
        graph = graph_from_state_pairs(trace, names)
    else:
        try:
            graph = parse_graph_json(raw)
        except json.JSONDecodeError as exc:
            # Recover object-shaped damaged exports from authoritative state
            # JSON. Extra program text is not part of the supported UTG format.
            if not raw.lstrip().startswith("{") or exc.msg == "Extra data":
                raise
            LOGGER.warning("Malformed MobileViews graph %s (%s); reading original screen/state pairs", graph_path, exc)
            graph = graph_from_state_pairs(trace, names)
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        raise ValueError(f"Invalid MobileViews UTG nodes: {graph_path}")
    package = graph.get("app_package")
    if not isinstance(package, str) or not package.strip():
        raise ValueError(f"Missing MobileViews app_package: {graph_path}")
    package = package.strip()
    visits = read_action_visits(trace / "actions.csv")
    metadata = ImageMetadataReader()
    trace_id = trace.relative_to(root).as_posix()
    seen_images: set[str] = set()
    seen_states: set[str] = set()
    rejected = 0
    nodes = graph["nodes"]
    if any(not isinstance(node, dict) for node in nodes):
        raise ValueError(f"Invalid MobileViews UTG node: {graph_path}")
    for node in sorted(nodes, key=lambda item: str(item.get("image", ""))):
        reference = Path(str(node.get("image") or ""))
        # Do not fall back from a crop/bad path to a similarly named screenshot.
        if len(reference.parts) != 2 or reference.parts[0] != "states":
            rejected += 1
            continue
        image_path = trace / reference
        # Directory containment is checked once above; scandir excludes symlinks.
        # Avoid resolving every ancestor of every image again over NFS.
        if reference.name not in names or _SCREEN_NAME.fullmatch(reference.name) is None:
            rejected += 1
            continue
        if str(image_path) in seen_images:
            continue
        state_id = str(node.get("state_str") or node.get("id") or "")
        foreground = str(node.get("package") or "")
        activity = str(node.get("activity") or "")
        foreground, activity = canonicalize_activity(f"{foreground}/{activity}")
        if not state_id or not foreground or not activity:
            rejected += 1
            continue
        external = foreground != package
        if external and not config.include_external:
            continue
        if config.deduplicate_by_state and state_id in seen_states:
            continue
        width = height = 0
        sha256 = ""
        if config.scan_image_metadata:
            # JSON width/height are not reliable in this export. Read the image.
            try:
                width, height = metadata.read_size(image_path)
            except (OSError, ValueError) as exc:
                LOGGER.warning("Skipping unreadable MobileViews screenshot %s: %s", image_path, exc)
                rejected += 1
                continue
        if config.scan_hashes:
            sha256 = metadata.sha256(image_path)
        xml_path = states_dir / f"window_dump_{image_path.stem[len('screen_'):]}.xml"
        xml_available = xml_path.name in names
        seen_images.add(str(image_path))
        seen_states.add(state_id)
        yield ScreenRecord(
            record_id=f"mobileviews:{image_path.relative_to(root).as_posix()}",
            source="mobileviews",
            image_path=str(image_path),
            xml_path=str(xml_path) if xml_available else "",
            package_name=package,
            activity_name=f"{foreground}/{activity}",
            structure_id=str(node.get("structure_str") or ""),
            state_id=state_id,
            trace_id=trace_id,
            is_external=external,
            # screen_N is an identifier, NOT a chronological observation index.
            sequence_index=-1,
            trace_visits=visits.get(state_id, ()),
            image_width=width,
            image_height=height,
            image_sha256=sha256,
        )
    if rejected:
        LOGGER.warning("MobileViews trace %s: rejected %d invalid/non-screen nodes", trace, rejected)


def graph_from_state_pairs(trace: Path, file_names: set[str]) -> dict:
    """Adapt incomplete graph exports from original full-screen/state pairs."""
    from ui_foundation.data.scanners import canonicalize_activity

    nodes = []
    for name in sorted(file_names):
        if _SCREEN_NAME.fullmatch(name) is None:
            continue
        state_name = f"state_{Path(name).stem[len('screen_'):]}.json"
        if state_name not in file_names:
            raise ValueError(f"Empty graph has a screenshot without state metadata: {trace / 'states' / name}")
        state_path = trace / "states" / state_name
        state = json.loads(state_path.read_text(encoding="utf-8"))
        package, activity = canonicalize_activity(str(state.get("foreground_activity") or ""))
        nodes.append({
            "image": f"states/{name}", "state_str": state.get("state_str", ""),
            "structure_str": state.get("state_str_content_free", ""),
            "package": package, "activity": activity,
        })
    # The original app-directory name retains ownership even when only external
    # launcher/permission screens were captured. Do not relabel them as the app.
    package = trace.name
    observed = {node["package"] for node in nodes}
    if package.endswith(".apk") and package not in observed:
        package = package.removesuffix(".apk")
    if _PACKAGE_NAME.fullmatch(package) is None:
        raise ValueError(f"Cannot establish app ownership for empty MobileViews graph: {trace}")
    return {"app_package": package, "nodes": nodes}
