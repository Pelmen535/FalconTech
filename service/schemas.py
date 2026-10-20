"""OpenAPI response descriptions only; do not filter or change runtime payloads."""
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class DocumentedResponse(BaseModel):
    model_config = ConfigDict(extra='allow')


class ErrorResponse(DocumentedResponse):
    detail: str | list[dict[str, Any]]


class HealthResponse(DocumentedResponse):
    status: str
    gallery_count: int
    model_loaded: bool
    dimension: int
    model_sha256: str
    recipe_sha256: str
    device: str
    storage: str
    busy: bool


class ModelResponse(DocumentedResponse):
    bbox_format: str = Field(description='XYWH: x, y, width, height, in pixels')
    confidence_scale: str = Field(description='raw_cosine or calibrated_acceptance_score; neither is a probability')
    loaded: bool | None = None
    dimension: int | None = None
    device: str | None = None
    model_sha256: str | None = None
    recipe_sha256: str | None = None
    embedding_sha256: str | None = None
    threshold: float | None = None
    backend: str | None = None
    rerank: str | None = None
    refusal_profile: str | None = None
    supports_ann: bool | None = None
    supports_explain: bool | None = None


class IndexResponse(DocumentedResponse):
    gallery_count: int
    active_mode: str
    exact_search: str
    ann: dict[str, Any]
    benchmark: dict[str, Any] | None


class GalleryItemResponse(DocumentedResponse):
    image_id: str
    bbox_xywh: tuple[float, float, float, float]
    created_at: str = Field(description='UTC ISO 8601 timestamp')
    image_url: str
    thumbnail_url: str


class GalleryResponse(DocumentedResponse):
    count: int
    items: list[GalleryItemResponse]


class ImportResponse(DocumentedResponse):
    added: int
    count: int


class DemoResponse(DocumentedResponse):
    available: bool
    image_url: str | None = None
    bbox_xywh: tuple[float, float, float, float] | None = None


class RankedItemResponse(DocumentedResponse):
    image_id: str
    rank: int
    cosine: float
    ranking_score: float | None = None
    image_url: str
    thumbnail_url: str


class SearchResponse(DocumentedResponse):
    accepted: bool
    confidence: float
    threshold: float
    confidence_scale: str
    candidate: RankedItemResponse | None = Field(description='Null when the model refuses identification; top-10 remains available')
    ranking: list[RankedItemResponse]
    gallery_count: int
    search_mode: str
    shortlist_seconds: float | None
    notes: list[str]
    model_sha256: str
    recipe_sha256: str
    seconds: float


class ExplainSideResponse(DocumentedResponse):
    heatmap_png_base64: str


class ExplainResponse(DocumentedResponse):
    gallery_id: str
    sides: dict[str, ExplainSideResponse]
    seconds: float


ERROR_DESCRIPTIONS = {
    400: 'Malformed request.', 401: 'Missing/incorrect X-API-Key when VREID_API_KEY is configured.',
    404: 'Requested observation or demo data not found.',
    409: 'Duplicate ID, gallery limit, empty/too small gallery, or incompatible operation.',
    413: 'Request or archive exceeds configured size limit.',
    422: 'Invalid fields, image, bbox, archive, or unsupported search mode.',
    501: 'Loaded model does not implement patch explanation.',
    503: 'Model is busy or resources unavailable; retry after Retry-After.',
}
REQUEST_HEADER = {'description': 'Server-generated identifier matching structured request/operation logs.',
                  'schema': {'type': 'string'}}


def responses(model=None, status=200, errors=(), authenticated=True, binary=False):
    success = {'description': 'Successful response.', 'headers': {'X-Request-ID': REQUEST_HEADER}}
    if model is not None:
        success['model'] = model
    elif binary:
        success['content'] = {media: {'schema': {'type': 'string', 'format': 'binary'}}
                              for media in ['image/jpeg', 'image/png', 'image/webp', 'application/octet-stream']}
    result = {status: success}
    for code in dict.fromkeys(([401] if authenticated else []) + list(errors)):
        result[code] = {'model': ErrorResponse, 'description': ERROR_DESCRIPTIONS[code],
                        'headers': {'X-Request-ID': REQUEST_HEADER}}
        if code == 503:
            result[code]['headers']['Retry-After'] = {'schema': {'type': 'string'}, 'description': 'Suggested delay in seconds.'}
    result[500] = {'model': ErrorResponse,
                   'description': 'Unexpected server failure; internal details are excluded.',
                   'headers': {'X-Request-ID': REQUEST_HEADER}}
    return result
