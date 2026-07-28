"""Frozen paper-protocol identifiers and non-secret evaluation provenance.

The scientific metric payload stays under ``raw_metrics``.  Provider/model
identity, prompt and image fingerprints, and request controls are written to a
separate per-case sidecar by :mod:`p3dbench.pipeline`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Iterable, Optional


PAPER_PROTOCOL_ID = "p3d-aaai27-paper-protocol-v1"
PAPER_QA_BANK_VERSION = 9
PAPER_CANONICAL_VIEW_COUNT = 4
PAPER_RENDER_RESOLUTION = 768
PAPER_RENDER_SAMPLES = 128
PAPER_RENDER_SEED = 42

_PAPER_DATASET_CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "contracts"
    / "p3d-aaai27-paper-protocol-v1.json"
)

# Golden hashes use the fixed fixtures in tests/test_paper_protocol.py. Keeping
# the same values in the research runner prevents prompt drift between the
# released reproduction code and the paper evaluator.
PAPER_PROMPT_GOLDEN_SHA256 = {
    "judge_semantic_only_fixture_v1": "f740f2fc696ba33b5e12696b64380f4d9c015ae424835d6cb392f7f0e5478440",
    "judge_visual_fixture_v1": "a355304fec2e82e45edef555f493e0e42672fbcbb7a048e8eb7458646b4cd31a",
    "qa_answer_fixture_v1": "378d6dfbcf61acb7e64c9d684461e234f236763c521a525b4e78b1ddfe35460c",
    "qa_answer_system_v1": "7793ecf39b464d66e512ef5c101b20bd7383decb5d557feb1849d89fffabe64d",
    "part_decompose_cadquery_fixture_v1": "9743d8c304fb87bd9164c004f35f6a91cef572d8fe64e3f0eeb6605c819a31b4",
    "part_decompose_openscad_fixture_v1": "a6d6ca621a03471ea761c9f08bef9cfaad3c3be8e09599999805c4d831ab434c",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON content independent of whitespace and dictionary key order."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def ordered_values_sha256(values: Iterable[str]) -> str:
    """Hash an ordered identity list using the frozen newline-delimited encoding."""
    return sha256_text("\n".join(str(value) for value in values))


def paper_dataset_contract() -> dict[str, Any]:
    """Return the checked-in dataset contract for the frozen paper protocol."""
    contract = json.loads(_PAPER_DATASET_CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("protocol_id") != PAPER_PROTOCOL_ID:
        raise ValueError(
            "Paper dataset contract protocol mismatch: "
            f"{contract.get('protocol_id')!r}"
        )
    if contract.get("schema_version") != 1:
        raise ValueError(
            "Unsupported paper dataset contract schema: "
            f"{contract.get('schema_version')!r}"
        )
    return contract


def paper_task_contract(task: str) -> dict[str, Any]:
    contract = paper_dataset_contract()
    try:
        return dict(contract["tasks"][task])
    except KeyError as exc:
        raise ValueError(f"No paper dataset contract for task {task!r}") from exc


def paper_manifest_provenance(task: str) -> dict[str, Any]:
    """Compact provenance record embedded in every materialized full-split row."""
    contract = paper_dataset_contract()
    task_contract = paper_task_contract(task)
    provenance = {
        "schema_version": contract["schema_version"],
        "protocol_id": contract["protocol_id"],
        "dataset_name": contract["dataset_name"],
        "dataset_revision": contract["dataset_revision"],
        "dataset_manifest_sha256": contract["dataset_manifest_sha256"],
        "inventory_reference_sha256": contract["inventory_reference_sha256"],
        "uids_path": task_contract["uids_path"],
        "uids_sha256": task_contract["uids_sha256"],
        "expected_count": task_contract["expected_count"],
        "ordered_source_ids_sha256": task_contract[
            "ordered_source_ids_sha256"
        ],
    }
    if task == "text-to-3d":
        qa_contract = contract["qa"]
        provenance.update({
            "qa_path": qa_contract["path"],
            "qa_source_sha256": qa_contract["source_sha256"],
            "qa_dataset_content_sha256": qa_contract[
                "normalized_content_sha256"
            ],
            "qa_bank_version": qa_contract["bank_version"],
        })
    return provenance


def validate_paper_source_ids(task: str, source_ids: Iterable[str]) -> dict[str, Any]:
    """Validate the exact source-ID universe and order for one paper task."""
    values = [str(value) for value in source_ids]
    task_contract = paper_task_contract(task)
    expected_count = int(task_contract["expected_count"])
    if len(values) != expected_count:
        raise ValueError(
            f"Paper dataset {task} requires {expected_count} source IDs; "
            f"got {len(values)}"
        )
    if any(not value for value in values):
        raise ValueError(f"Paper dataset {task} has an empty source ID")
    if len(set(values)) != len(values):
        raise ValueError(f"Paper dataset {task} has duplicate source IDs")
    digest = ordered_values_sha256(values)
    if digest != task_contract["ordered_source_ids_sha256"]:
        raise ValueError(
            f"Paper dataset {task} source-ID order/content digest mismatch: "
            f"expected {task_contract['ordered_source_ids_sha256']}, got {digest}"
        )
    return {
        "expected_count": expected_count,
        "ordered_source_ids_sha256": digest,
    }


def normalize_frozen_qa_questions(questions: Iterable[dict]) -> list[dict]:
    """Add only the two frozen packaging fields absent from the source release."""
    normalized: list[dict] = []
    for raw_question in questions:
        question = dict(raw_question)
        split = (
            "param"
            if str(question.get("qid", "")).startswith("param")
            else "semantic"
        )
        question.setdefault("split", split)
        question.setdefault(
            "source_text_level",
            "parametric_detail" if split == "param" else "detailed",
        )
        normalized.append(question)
    return normalized


def qa_questions_content_sha256(questions: Iterable[dict]) -> str:
    return canonical_json_sha256(list(questions))


def qa_dataset_content_sha256(rows: Iterable[dict]) -> str:
    """Hash normalized QA rows in authoritative source order."""
    normalized_rows = [
        {
            "uid": str(row.get("uid") or ""),
            "questions": normalize_frozen_qa_questions(
                row.get("questions") or []
            ),
        }
        for row in rows
    ]
    return canonical_json_sha256(normalized_rows)


def paper_qa_source_contract() -> dict[str, Any]:
    contract = paper_dataset_contract()
    qa = contract["qa"]
    return {
        "protocol_id": contract["protocol_id"],
        "dataset_name": contract["dataset_name"],
        "dataset_revision": contract["dataset_revision"],
        "dataset_manifest_sha256": contract["dataset_manifest_sha256"],
        "path": qa["path"],
        "source_sha256": qa["source_sha256"],
        "normalized_content_sha256": qa["normalized_content_sha256"],
        "bank_version": qa["bank_version"],
    }


def validate_materialized_qa_bank_contract(
    qa_bank: dict,
    *,
    expected_uid: str,
) -> None:
    """Validate source provenance and self-consistency of one materialized bank."""
    if str(qa_bank.get("uid") or "") != str(expected_uid):
        raise ValueError(
            f"QA bank UID mismatch: expected {expected_uid}, "
            f"got {qa_bank.get('uid')!r}"
        )
    if qa_bank.get("qa_bank_version") != PAPER_QA_BANK_VERSION:
        raise ValueError(
            f"Paper protocol requires QA bank v{PAPER_QA_BANK_VERSION}"
        )
    expected_source = paper_qa_source_contract()
    if qa_bank.get("source_contract") != expected_source:
        raise ValueError("QA bank source SHA/revision contract mismatch")
    questions = qa_bank.get("questions")
    if not isinstance(questions, list):
        raise ValueError("QA bank questions must be a list")
    actual_content_sha = qa_questions_content_sha256(questions)
    if qa_bank.get("content_sha256") != actual_content_sha:
        raise ValueError("QA bank content digest mismatch")
    expected_dataset_sha = expected_source["normalized_content_sha256"]
    if qa_bank.get("dataset_content_sha256") != expected_dataset_sha:
        raise ValueError("QA bank dataset digest mismatch")


def validate_paper_materialized_manifest(
    task: str,
    rows: Iterable[dict],
    *,
    split: str,
    root: Optional[Path] = None,
) -> dict[str, Any]:
    """Validate the exact materialized case and QA universe before model calls."""
    if split != "full":
        raise ValueError("Paper protocol requires split='full'")
    materialized = list(rows)
    task_contract = paper_task_contract(task)
    expected_count = int(task_contract["expected_count"])
    if len(materialized) != expected_count:
        raise ValueError(
            f"Paper protocol {task} requires exactly {expected_count} "
            f"materialized cases; got {len(materialized)}"
        )

    expected_ids = [
        f"{task_contract['case_id_prefix']}{index:06d}"
        for index in range(expected_count)
    ]
    actual_ids = [str(row.get("id") or "") for row in materialized]
    if actual_ids != expected_ids:
        raise ValueError(
            f"Paper protocol {task} case ID/order mismatch"
        )

    wrong_task = [
        row.get("id")
        for row in materialized
        if row.get("task") != task
    ]
    wrong_split = [
        row.get("id")
        for row in materialized
        if row.get("split") != split
    ]
    if wrong_task or wrong_split:
        raise ValueError(
            "Paper protocol manifest task/split mismatch: "
            f"wrong_task={wrong_task[:20]}, wrong_split={wrong_split[:20]}"
        )

    source_ids: list[str] = []
    expected_provenance = paper_manifest_provenance(task)
    provenance_mismatches: list[str] = []
    for row in materialized:
        metadata = row.get("metadata") or {}
        source_ids.append(str(metadata.get("source_id") or ""))
        if metadata.get("paper_dataset_contract") != expected_provenance:
            provenance_mismatches.append(str(row.get("id") or ""))
    validate_paper_source_ids(task, source_ids)
    if provenance_mismatches:
        raise ValueError(
            "Paper materialized manifest source SHA/revision contract "
            f"mismatch for cases {provenance_mismatches[:20]}"
        )

    qa_dataset_sha = None
    if task == "text-to-3d":
        if root is None:
            raise ValueError(
                "Paper Text-to-3D manifest validation requires its data root"
            )
        resolved_root = Path(root).resolve()
        qa_rows: list[dict] = []
        for row, source_id in zip(materialized, source_ids):
            target = row.get("target") or {}
            render_paths = list(target.get("render_paths") or [])
            if len(render_paths) != PAPER_CANONICAL_VIEW_COUNT:
                raise ValueError(
                    f"Paper TextDesc case {row.get('id')} requires exactly "
                    f"{PAPER_CANONICAL_VIEW_COUNT} GT renders"
                )
            render_contract = (
                (row.get("metadata") or {}).get(
                    "paper_gt_render_contract"
                )
                or {}
            )
            fixed_render_contract = {
                "renderer": "blender_clay",
                "view_count": PAPER_CANONICAL_VIEW_COUNT,
                "resolution": PAPER_RENDER_RESOLUTION,
                "samples": PAPER_RENDER_SAMPLES,
                "seed": PAPER_RENDER_SEED,
            }
            if any(
                render_contract.get(key) != value
                for key, value in fixed_render_contract.items()
            ):
                raise ValueError(
                    f"Paper TextDesc case {row.get('id')} GT render "
                    "profile mismatch"
                )
            mesh_relative = target.get("mesh_path")
            if not mesh_relative:
                raise ValueError(
                    f"Paper TextDesc case {row.get('id')} has no GT mesh"
                )
            mesh_path = (resolved_root / str(mesh_relative)).resolve()
            try:
                mesh_path.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(
                    f"GT mesh path escapes the full data root: {mesh_relative}"
                ) from exc
            if (
                not mesh_path.is_file()
                or render_contract.get("source_mesh_sha256")
                != file_fingerprint(str(mesh_path))["sha256"]
            ):
                raise ValueError(
                    f"Paper TextDesc case {row.get('id')} GT mesh/render "
                    "binding mismatch"
                )
            contract_views = render_contract.get("views") or []
            if len(contract_views) != PAPER_CANONICAL_VIEW_COUNT:
                raise ValueError(
                    f"Paper TextDesc case {row.get('id')} GT render "
                    "contract is incomplete"
                )
            for index, relative_render in enumerate(render_paths):
                render_path = (
                    resolved_root / str(relative_render)
                ).resolve()
                try:
                    render_path.relative_to(resolved_root)
                except ValueError as exc:
                    raise ValueError(
                        "GT render path escapes the full data root: "
                        f"{relative_render}"
                    ) from exc
                expected_view = contract_views[index]
                if (
                    not render_path.is_file()
                    or expected_view.get("path") != relative_render
                    or expected_view.get("sha256")
                    != file_fingerprint(str(render_path))["sha256"]
                    or _png_dimensions(render_path) != (
                        PAPER_RENDER_RESOLUTION,
                        PAPER_RENDER_RESOLUTION,
                    )
                ):
                    raise ValueError(
                        f"Paper TextDesc case {row.get('id')} GT render "
                        f"{index} binding/profile mismatch"
                    )
            relative = (row.get("target") or {}).get("qa_bank_path")
            if not relative:
                raise ValueError(
                    f"Paper Text-to-3D case {row.get('id')} has no QA bank"
                )
            qa_path = (resolved_root / str(relative)).resolve()
            try:
                qa_path.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(
                    f"QA bank path escapes the full data root: {relative}"
                ) from exc
            if not qa_path.is_file():
                raise ValueError(f"Paper QA bank is missing: {qa_path}")
            qa_bank = json.loads(qa_path.read_text(encoding="utf-8"))
            validate_materialized_qa_bank_contract(
                qa_bank,
                expected_uid=source_id,
            )
            qa_rows.append({
                "uid": source_id,
                "questions": qa_bank["questions"],
            })
        qa_dataset_sha = qa_dataset_content_sha256(qa_rows)
        expected_qa_sha = paper_dataset_contract()["qa"][
            "normalized_content_sha256"
        ]
        if qa_dataset_sha != expected_qa_sha:
            raise ValueError(
                "Materialized QA bank universe content/order digest mismatch: "
                f"expected {expected_qa_sha}, got {qa_dataset_sha}"
            )

    contract = paper_dataset_contract()
    result = {
        "task_profile": task,
        "split": split,
        "expected_count": expected_count,
        "ordered_ids_sha256": ordered_values_sha256(expected_ids),
        "ordered_source_ids_sha256": task_contract[
            "ordered_source_ids_sha256"
        ],
        "uids_sha256": task_contract["uids_sha256"],
        "dataset_revision": contract["dataset_revision"],
        "dataset_manifest_sha256": contract["dataset_manifest_sha256"],
    }
    if qa_dataset_sha is not None:
        result.update({
            "qa_source_sha256": contract["qa"]["source_sha256"],
            "qa_dataset_content_sha256": qa_dataset_sha,
            "qa_bank_version": PAPER_QA_BANK_VERSION,
        })
    return result


def file_fingerprint(path: str) -> dict[str, Any]:
    p = Path(path)
    return {
        "name": p.name,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "bytes": p.stat().st_size,
    }


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])


_PRIVATE_TRACE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "access_token",
    "refresh_token",
    "bearer_token",
    "token",
    "base_url",
    "credential",
    "credentials",
    "cookie",
    "cookies",
    "endpoint",
    "header",
    "headers",
    "password",
    "secret",
    "url",
}


def _is_private_trace_key(key: Any) -> bool:
    normalized = str(key).lower().replace("-", "_")
    if normalized in _PRIVATE_TRACE_KEYS:
        return True
    return normalized.endswith((
        "_api_key",
        "_access_token",
        "_auth_token",
        "_bearer_token",
        "_refresh_token",
        "_credential",
        "_credentials",
        "_cookie",
        "_cookies",
        "_endpoint",
        "_header",
        "_headers",
        "_password",
        "_secret",
        "_url",
    ))


def _secret_free_trace_value(value: Any) -> Any:
    """Recursively remove endpoint and credential-bearing fields from traces."""
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, nested in value.items():
            if _is_private_trace_key(key):
                continue
            clean[str(key)] = _secret_free_trace_value(nested)
        return clean
    if isinstance(value, (list, tuple)):
        return [_secret_free_trace_value(item) for item in value]
    return value


def model_call_trace(
    *,
    kind: str,
    client: Any,
    response: Any,
    prompt: str,
    images: Iterable[str],
    system: Optional[str] = None,
    request_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a secret-free provenance record for one evaluator model call."""
    cfg = getattr(client, "cfg", None)
    requested_model = getattr(cfg, "model", None)
    returned_model = getattr(response, "model", None)
    provider = getattr(cfg, "provider", None)
    trace = {
        "kind": kind,
        "protocol_id": PAPER_PROTOCOL_ID,
        "prompt_sha256": sha256_text(prompt),
        "system_sha256": sha256_text(system) if system else None,
        "images": [file_fingerprint(str(path)) for path in images],
        "request": {
            "provider": provider,
            "model_key": getattr(cfg, "name", None),
            "requested_model": requested_model,
            **(request_overrides or {}),
        },
        "response": {
            "returned_model": returned_model,
            "finish_reason": getattr(response, "finish_reason", None),
            "model_identity_matches": (
                returned_model == requested_model
                if returned_model is not None and requested_model is not None
                else None
            ),
        },
    }
    return _secret_free_trace_value(trace)


def append_call_trace(shared: dict[str, Any], trace: dict[str, Any]) -> None:
    shared.setdefault("evaluation_calls", []).append(trace)
