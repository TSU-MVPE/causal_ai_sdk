"""Endpoint-to-schema registry for static contract verification."""

from typing import Tuple

from causal_ai_sdk.contracts.requests import (
    CDLingamRunRequestContract,
    CDMatchSetRequestContract,
    CDMulticaMatchStartRequestContract,
    CDMulticaRunRequestContract,
    CDTrackrMatchStartRequestContract,
    CDTrackrRunRequestContract,
    DATaskRequestContract,
    KGAddRequestContract,
    KGUploadURLRequestContract,
)
from causal_ai_sdk.contracts.types import EndpointContract

CONTRACTS: Tuple[EndpointContract, ...] = (
    EndpointContract(
        name="kg.init_session",
        method="POST",
        path="/kg/init",
        openapi_response_schema="KGInitResponse",
    ),
    EndpointContract(
        name="kg.get_upload_url",
        method="POST",
        path="/kg/upload-url/{uuid}",
        request_model=KGUploadURLRequestContract,
        openapi_request_schema="KGUploadUrlRequest",
        openapi_response_schema="KGUploadUrlResponse",
    ),
    EndpointContract(
        name="kg.add_kg",
        method="POST",
        path="/kg/add/{uuid}",
        request_model=KGAddRequestContract,
        openapi_request_schema="KGAddRequest",
        notes="SDK adapts response {kg_id} into dict; OpenAPI validates request.",
    ),
    EndpointContract(
        name="kg.list_kg",
        method="GET",
        path="/kg/list/{uuid}",
        openapi_response_schema="KGListItem",
        notes="SDK returns list of dicts; OpenAPI validates response.",
    ),
    EndpointContract(
        name="kg.get_kg",
        method="GET",
        path="/kg/graph/{uuid}",
        openapi_response_schema="KGGraphPayload",
        notes="SDK returns response['kg'] dict; OpenAPI validates response.",
    ),
    EndpointContract(
        name="kg.delete_session",
        method="DELETE",
        path="/kg/graph/{uuid}",
        notes="SDK delete_session calls this endpoint and ignores response payload.",
    ),
    EndpointContract(
        name="cd.get_upload_url",
        method="POST",
        path="/cd/upload-url/{uuid}",
        openapi_response_schema="CDUploadUrlResponse",
    ),
    EndpointContract(
        name="cd.start_multica_matching",
        method="POST",
        path="/cd/multica/match/{uuid}",
        request_model=CDMulticaMatchStartRequestContract,
        openapi_request_schema="CDMulticaMatchStartRequest",
        openapi_response_schema="CDMatchStartResponse",
    ),
    EndpointContract(
        name="cd.set_multica_matching",
        method="POST",
        path="/cd/multica/match/{uuid}/set",
        request_model=CDMatchSetRequestContract,
        openapi_request_schema="CDMulticaMatchSetRequest",
        notes="SDK set_multica_matching currently ignores success response payload.",
    ),
    EndpointContract(
        name="cd.get_multica_matching",
        method="GET",
        path="/cd/multica/match/{uuid}",
        openapi_response_schema="CDMatchGetResponse",
        response_sdk_only_fields=("error",),
        response_sdk_only_reasons={
            "error": "SDK polling flow inspects error message when matching status is failed."
        },
        notes="SDK keeps optional 'error' field for failed/polling states.",
    ),
    EndpointContract(
        name="cd.run_multica",
        method="POST",
        path="/cd/multica/run/{uuid}",
        request_model=CDMulticaRunRequestContract,
        openapi_request_schema="CDMulticaRunRequest",
        openapi_response_schema="CDQueuedResponse",
    ),
    EndpointContract(
        name="cd.start_trackr_matching",
        method="POST",
        path="/cd/trackr/match/{uuid}",
        request_model=CDTrackrMatchStartRequestContract,
        openapi_request_schema="CDTrackrMatchStartRequest",
        openapi_response_schema="CDMatchStartResponse",
    ),
    EndpointContract(
        name="cd.set_trackr_matching",
        method="POST",
        path="/cd/trackr/match/{uuid}/set",
        request_model=CDMatchSetRequestContract,
        openapi_request_schema="CDTrackrMatchSetRequest",
        notes="SDK set_trackr_matching currently ignores success response payload.",
    ),
    EndpointContract(
        name="cd.get_trackr_matching",
        method="GET",
        path="/cd/trackr/match/{uuid}",
        openapi_response_schema="CDMatchGetResponse",
        response_sdk_only_fields=("error",),
        response_sdk_only_reasons={
            "error": "SDK polling flow inspects error message when matching status is failed."
        },
        notes="SDK keeps optional 'error' field for failed/polling states.",
    ),
    EndpointContract(
        name="cd.run_trackr",
        method="POST",
        path="/cd/trackr/run/{uuid}",
        request_model=CDTrackrRunRequestContract,
        openapi_request_schema="CDTrackrRunRequest",
        openapi_response_schema="CDQueuedResponse",
    ),
    EndpointContract(
        name="cd.run_lingam",
        method="POST",
        path="/cd/lingam/run/{uuid}",
        request_model=CDLingamRunRequestContract,
        openapi_request_schema="CDLingamRunRequest",
        openapi_response_schema="CDQueuedResponse",
    ),
    EndpointContract(
        name="cd.get_task_status",
        method="GET",
        path="/cd/status/{uuid}",
        openapi_response_schema="CDStatusResponse",
    ),
    EndpointContract(
        name="cd.get_task_result",
        method="GET",
        path="/cd/result/{uuid}",
        openapi_response_schema="CDResultResponse",
    ),
    EndpointContract(
        name="cd.delete_task",
        method="DELETE",
        path="/cd/task/{uuid}",
        notes="SDK delete_task calls this endpoint and ignores response payload.",
    ),
    EndpointContract(
        name="cd.delete_matching",
        method="DELETE",
        path="/cd/match/{uuid}",
        notes="SDK delete_matching calls this endpoint and ignores response payload.",
    ),
    # Decision Analysis (DA)
    EndpointContract(
        name="da.run_explain",
        method="POST",
        path="/da/explain/{uuid}",
        request_model=DATaskRequestContract,
        openapi_request_schema="DATaskRequest",
        openapi_response_schema="DAQueuedResponse",
    ),
    EndpointContract(
        name="da.run_enumerate",
        method="POST",
        path="/da/enumerate/{uuid}",
        request_model=DATaskRequestContract,
        openapi_request_schema="DATaskRequest",
        openapi_response_schema="DAQueuedResponse",
    ),
    EndpointContract(
        name="da.get_task_status",
        method="GET",
        path="/da/status/{uuid}",
        openapi_response_schema="DAStatusResponse",
    ),
    EndpointContract(
        name="da.get_task_result",
        method="GET",
        path="/da/result/{uuid}",
        openapi_response_schema="DAResultResponse",
    ),
    EndpointContract(
        name="da.delete_task",
        method="DELETE",
        path="/da/task/{uuid}",
        notes="SDK delete_task calls this endpoint and ignores response payload.",
    ),
)
