export interface DraftVersionResponse {
  version_id: number;
  topic: string;
  outline: string;
  research_notes: string;
  draft: string;
  review: string;
  revised: string;
  created_at: string;
}

export interface VersionsResponse {
  versions: DraftVersionResponse[];
}

export interface VersionDetailResponse {
  version: DraftVersionResponse;
}

export interface DeleteVersionResponse {
  deleted: boolean;
}

export interface VersionDiffResponse {
  from_version_id: number;
  to_version_id: number;
  field: string;
  diff: string;
}
