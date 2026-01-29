export interface HealthDetailResponse {
  status: string;
  memory_mode: string;
  storage_path: string;
  qdrant: {
    enabled: boolean;
    available: boolean;
    error?: string | null;
    url: string;
    collection: string;
    dim: number;
    distance: string;
  };
  embedding: {
    provider: string;
    model: string;
    dim?: number | null;
    mode: string;
    available: boolean;
    error?: string | null;
  };
}
