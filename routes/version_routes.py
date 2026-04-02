# version_routes.py - Contract Version History
"""
Track clause changes, customizations, and variant switches over time.
Uses the existing `contract_versions` table.

Endpoints:
- GET  /{id}/versions                  — List all versions
- GET  /{id}/versions/{v}              — Get specific version detail
- POST /{id}/versions                  — Create manual snapshot
- POST /{id}/versions/{v}/restore      — Restore to a previous version
- GET  /{id}/versions/{v1}/compare/{v2} — Diff two versions
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from config import DB_CONFIG
import psycopg2
from psycopg2.extras import RealDictCursor

router = APIRouter(prefix="/api/contracts", tags=["version-history"])


# ==================== PYDANTIC MODELS ====================

class CreateVersionRequest(BaseModel):
    change_summary: Optional[str] = "Manual snapshot"


class VersionResponse(BaseModel):
    id: int
    contract_id: str
    version_number: int
    change_summary: Optional[str]
    changed_by: Optional[str]
    created_at: datetime
    clause_count: int


class VersionDetailResponse(VersionResponse):
    content: Dict[str, Any]


class VersionDiff(BaseModel):
    version_a: int
    version_b: int
    changes: List[Dict[str, Any]]
    summary: str


# ==================== HELPERS ====================

def get_db():
    return psycopg2.connect(**DB_CONFIG)


def create_version_snapshot(
    contract_id: str,
    change_summary: str = "Snapshot",
    changed_by: str = None,
) -> dict:
    """
    Create a version snapshot of the contract's current state.
    Stores full clause configuration + parameter values.

    This function is PUBLIC — imported by other routes for auto-versioning.
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Verify contract exists
            cur.execute("SELECT id FROM contracts WHERE id = %s", (contract_id,))
            if not cur.fetchone():
                return None  # Silently skip for auto-versioning hooks

            # Get current clause state
            cur.execute("""
                SELECT clause_id, clause_type, variant, sequence,
                       is_mandatory, is_active, is_customized, overridden_text
                FROM contract_clauses
                WHERE contract_id = %s
                ORDER BY sequence
            """, (contract_id,))
            clauses = [dict(r) for r in cur.fetchall()]

            # Get parameter values
            cur.execute("""
                SELECT parameter_id, value_text, value_integer,
                       value_decimal, value_date, value_currency
                FROM contract_parameters
                WHERE contract_id = %s
            """, (contract_id,))
            params = {}
            for row in cur.fetchall():
                pid = row["parameter_id"]
                val = (row["value_text"] or row["value_integer"] or
                       row["value_decimal"] or
                       (str(row["value_date"]) if row["value_date"] else None) or
                       (json.dumps(row["value_currency"]) if row["value_currency"] else None))
                if val is not None:
                    params[pid] = val

            # Get next version number
            cur.execute("""
                SELECT COALESCE(MAX(version_number), 0) + 1 AS next_version
                FROM contract_versions
                WHERE contract_id = %s
            """, (contract_id,))
            next_version = cur.fetchone()["next_version"]

            # Build snapshot content
            content = {
                "clauses": clauses,
                "parameters": params,
                "metadata": {
                    "snapshot_at": datetime.now().isoformat(),
                    "active_clause_count": sum(1 for c in clauses if c.get("is_active")),
                    "total_clause_count": len(clauses),
                    "parameter_count": len(params),
                }
            }

            # Insert version
            cur.execute("""
                INSERT INTO contract_versions
                    (contract_id, version_number, content, change_summary, changed_by)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING *
            """, (
                contract_id,
                next_version,
                json.dumps(content, default=str),
                change_summary,
                changed_by,
            ))
            version = cur.fetchone()
            conn.commit()

        return dict(version)
    finally:
        conn.close()


# ==================== ROUTES ====================

@router.get("/{contract_id}/versions", response_model=List[VersionResponse])
async def list_versions(contract_id: str):
    """List all version snapshots for a contract, newest first."""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM contracts WHERE id = %s", (contract_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="Contract not found")

            cur.execute("""
                SELECT id, contract_id, version_number, change_summary,
                       changed_by, created_at, content
                FROM contract_versions
                WHERE contract_id = %s
                ORDER BY version_number DESC
            """, (contract_id,))
            versions = cur.fetchall()

        return [
            {
                **{k: v for k, v in dict(v).items() if k != "content"},
                "contract_id": str(v["contract_id"]),
                "changed_by": str(v["changed_by"]) if v["changed_by"] else None,
                "clause_count": v["content"].get("metadata", {}).get("active_clause_count", 0)
                    if v.get("content") else 0,
            }
            for v in versions
        ]
    finally:
        conn.close()


@router.get("/{contract_id}/versions/{version_number}", response_model=VersionDetailResponse)
async def get_version(contract_id: str, version_number: int):
    """Get detailed contents of a specific version."""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM contract_versions
                WHERE contract_id = %s AND version_number = %s
            """, (contract_id, version_number))
            version = cur.fetchone()

            if not version:
                raise HTTPException(
                    status_code=404,
                    detail=f"Version {version_number} not found for this contract"
                )

        return {
            **dict(version),
            "contract_id": str(version["contract_id"]),
            "changed_by": str(version["changed_by"]) if version["changed_by"] else None,
            "clause_count": version["content"].get("metadata", {}).get("active_clause_count", 0)
                if version.get("content") else 0,
        }
    finally:
        conn.close()


@router.post("/{contract_id}/versions", response_model=VersionResponse)
async def create_version(contract_id: str, request: CreateVersionRequest = None):
    """
    Create a manual version snapshot of the contract's current state.
    Useful before making significant changes.
    """
    summary = request.change_summary if request else "Manual snapshot"
    version = create_version_snapshot(contract_id, summary)

    if not version:
        raise HTTPException(status_code=404, detail="Contract not found")

    return {
        **{k: v for k, v in version.items() if k != "content"},
        "contract_id": str(version["contract_id"]),
        "changed_by": str(version["changed_by"]) if version["changed_by"] else None,
        "clause_count": version["content"].get("metadata", {}).get("active_clause_count", 0)
            if version.get("content") else 0,
    }


@router.post("/{contract_id}/versions/{version_number}/restore")
async def restore_version(contract_id: str, version_number: int):
    """
    Restore a contract to a previous version.

    This will:
    1. Create a snapshot of the current state (as a safety measure)
    2. Set clause active/inactive states to match the target version
    3. Restore any customized text from the target version
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get the target version
            cur.execute("""
                SELECT * FROM contract_versions
                WHERE contract_id = %s AND version_number = %s
            """, (contract_id, version_number))
            target = cur.fetchone()

            if not target:
                raise HTTPException(
                    status_code=404,
                    detail=f"Version {version_number} not found"
                )

            # Safety snapshot of current state
            create_version_snapshot(
                contract_id,
                f"Auto-snapshot before restore to v{version_number}"
            )

            target_content = target["content"]
            target_clauses = target_content.get("clauses", [])

            # Build maps from target version
            active_ids = set()
            customized_texts = {}
            for clause in target_clauses:
                cid = clause["clause_id"]
                if clause.get("is_active"):
                    active_ids.add(cid)
                if clause.get("is_customized") and clause.get("overridden_text"):
                    customized_texts[cid] = clause["overridden_text"]

            # Step 1: Deactivate all
            cur.execute("""
                UPDATE contract_clauses
                SET is_active = false, updated_at = NOW()
                WHERE contract_id = %s
            """, (contract_id,))

            # Step 2: Activate the target version's clauses
            if active_ids:
                cur.execute("""
                    UPDATE contract_clauses
                    SET is_active = true, updated_at = NOW()
                    WHERE contract_id = %s AND clause_id = ANY(%s)
                """, (contract_id, list(active_ids)))
            activated = len(active_ids)

            # Step 3: Restore customized texts
            restored_customs = 0
            for cid, text in customized_texts.items():
                cur.execute("""
                    UPDATE contract_clauses
                    SET overridden_text = %s, is_customized = true, updated_at = NOW()
                    WHERE contract_id = %s AND clause_id = %s
                """, (text, contract_id, cid))
                restored_customs += cur.rowcount

            conn.commit()

        # Create a post-restore snapshot
        create_version_snapshot(
            contract_id,
            f"Restored to version {version_number}"
        )

        return {
            "message": f"Contract restored to version {version_number}",
            "contract_id": contract_id,
            "restored_version": version_number,
            "clauses_activated": activated,
            "customizations_restored": restored_customs,
        }
    finally:
        conn.close()


@router.get("/{contract_id}/versions/{v1}/compare/{v2}", response_model=VersionDiff)
async def compare_versions(contract_id: str, v1: int, v2: int):
    """
    Compare two versions clause-by-clause.
    Returns a list of differences (added, removed, changed clauses).
    """
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT version_number, content FROM contract_versions
                WHERE contract_id = %s AND version_number IN (%s, %s)
                ORDER BY version_number
            """, (contract_id, v1, v2))
            versions = {r["version_number"]: r["content"] for r in cur.fetchall()}

        if v1 not in versions:
            raise HTTPException(status_code=404, detail=f"Version {v1} not found")
        if v2 not in versions:
            raise HTTPException(status_code=404, detail=f"Version {v2} not found")

        clauses_a = {c["clause_id"]: c for c in versions[v1].get("clauses", [])}
        clauses_b = {c["clause_id"]: c for c in versions[v2].get("clauses", [])}

        all_clause_ids = set(clauses_a.keys()) | set(clauses_b.keys())
        changes = []

        for cid in sorted(all_clause_ids):
            a = clauses_a.get(cid)
            b = clauses_b.get(cid)

            if a and not b:
                changes.append({
                    "clause_id": cid,
                    "change_type": "removed",
                    "clause_type": a.get("clause_type"),
                    "details": f"Clause {cid} was removed in v{v2}",
                })
            elif b and not a:
                changes.append({
                    "clause_id": cid,
                    "change_type": "added",
                    "clause_type": b.get("clause_type"),
                    "details": f"Clause {cid} was added in v{v2}",
                })
            elif a and b:
                diffs = []

                # Check active state
                if a.get("is_active") != b.get("is_active"):
                    old_state = "active" if a.get("is_active") else "inactive"
                    new_state = "active" if b.get("is_active") else "inactive"
                    diffs.append(f"State: {old_state} → {new_state}")

                # Check variant
                if a.get("variant") != b.get("variant"):
                    diffs.append(f"Variant: {a.get('variant')} → {b.get('variant')}")

                # Check customization
                if a.get("is_customized") != b.get("is_customized"):
                    if b.get("is_customized"):
                        diffs.append("Customized text applied")
                    else:
                        diffs.append("Customization reverted")

                # Check overridden text
                if a.get("overridden_text") != b.get("overridden_text"):
                    if b.get("overridden_text") and a.get("overridden_text"):
                        diffs.append("Custom text modified")
                    elif b.get("overridden_text"):
                        diffs.append("Custom text added")

                if diffs:
                    changes.append({
                        "clause_id": cid,
                        "change_type": "modified",
                        "clause_type": a.get("clause_type"),
                        "details": "; ".join(diffs),
                    })

        summary = (
            f"Comparing v{v1} → v{v2}: "
            f"{sum(1 for c in changes if c['change_type'] == 'added')} added, "
            f"{sum(1 for c in changes if c['change_type'] == 'removed')} removed, "
            f"{sum(1 for c in changes if c['change_type'] == 'modified')} modified"
        )

        return {
            "version_a": v1,
            "version_b": v2,
            "changes": changes,
            "summary": summary,
        }
    finally:
        conn.close()
