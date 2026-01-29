from __future__ import annotations

from difflib import unified_diff

from fastapi import APIRouter, Depends, HTTPException, Query

from ...models.schemas import (
    DeleteVersionResponse,
    DraftVersionResponse,
    VersionDetailResponse,
    VersionDiffResponse,
    VersionsResponse,
)
from ..deps import AppServices, get_services

router = APIRouter(tags=["versions"])


@router.get("/versions", response_model=VersionsResponse)
def list_versions(
    limit: int | None = Query(default=None, ge=1, le=200),
    services: AppServices = Depends(get_services),
) -> VersionsResponse:
    try:
        versions = services.storage.list_draft_versions(limit=limit)
        payload = [
            DraftVersionResponse(
                version_id=item.version_id,
                topic=item.topic,
                outline=item.outline,
                research_notes=item.research_notes,
                draft=item.draft,
                review=item.review,
                revised=item.revised,
                created_at=item.created_at,
            )
            for item in versions
        ]
        return VersionsResponse(versions=payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="List versions failed") from exc


@router.get("/versions/{version_id}", response_model=VersionDetailResponse)
def get_version(
    version_id: int,
    services: AppServices = Depends(get_services),
) -> VersionDetailResponse:
    try:
        version = services.storage.get_draft_version(version_id)
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
        payload = DraftVersionResponse(
            version_id=version.version_id,
            topic=version.topic,
            outline=version.outline,
            research_notes=version.research_notes,
            draft=version.draft,
            review=version.review,
            revised=version.revised,
            created_at=version.created_at,
        )
        return VersionDetailResponse(version=payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Get version failed") from exc


@router.delete("/versions/{version_id}", response_model=DeleteVersionResponse)
def delete_version(
    version_id: int,
    services: AppServices = Depends(get_services),
) -> DeleteVersionResponse:
    try:
        deleted = services.storage.delete_draft_version(version_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Version not found")
        return DeleteVersionResponse(deleted=True)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Delete version failed") from exc


@router.get("/versions/{version_id}/diff", response_model=VersionDiffResponse)
def diff_versions(
    version_id: int,
    compare_to: int | None = Query(default=None, description="对比的版本ID"),
    field: str = Query(default="revised", description="比较字段"),
    services: AppServices = Depends(get_services),
) -> VersionDiffResponse:
    try:
        current = services.storage.get_draft_version(version_id)
        if not current:
            raise HTTPException(status_code=404, detail="Version not found")

        if compare_to is None:
            previous = services.storage.get_previous_version(version_id)
            if not previous:
                raise HTTPException(status_code=404, detail="No previous version to diff")
        else:
            previous = services.storage.get_draft_version(compare_to)
            if not previous:
                raise HTTPException(status_code=404, detail="Compare version not found")

        allowed_fields = {
            "outline": ("Outline", current.outline, previous.outline),
            "research_notes": ("Research Notes", current.research_notes, previous.research_notes),
            "draft": ("Draft", current.draft, previous.draft),
            "review": ("Review", current.review, previous.review),
            "revised": ("Revised", current.revised, previous.revised),
        }
        if field not in allowed_fields:
            raise HTTPException(status_code=400, detail="Unsupported diff field")

        title, current_text, previous_text = allowed_fields[field]
        diff_lines = unified_diff(
            previous_text.splitlines(),
            current_text.splitlines(),
            fromfile=f"{title} (v{previous.version_id})",
            tofile=f"{title} (v{current.version_id})",
            lineterm="",
        )
        diff_text = "\n".join(diff_lines)
        return VersionDiffResponse(
            from_version_id=previous.version_id,
            to_version_id=current.version_id,
            field=field,
            diff=diff_text,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Diff versions failed") from exc
