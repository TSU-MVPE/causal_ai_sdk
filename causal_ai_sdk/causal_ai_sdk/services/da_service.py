"""Decision Analysis service."""

from typing import Any, Dict, List, Optional

from causal_ai_sdk.contracts import get_contract
from causal_ai_sdk.contracts.requests import DATaskRequestContract
from causal_ai_sdk.services.base import BaseService
from causal_ai_sdk.utils.polling import poll_until_ready_or_fail


class DAService(BaseService):
    """Decision Analysis service for explain and enumerate tasks.

    This service provides methods to run DA tasks (explain or enumerate),
    poll status, fetch result URL, and delete tasks. All operations are
    session- and task-scoped (status/result/delete require task_id as query param).
    """

    @staticmethod
    def _contract_path(name: str, **path_params: str) -> str:
        """Build endpoint path from the contract registry.

        Args:
            name (str): Contract registry key (e.g. da.run_explain).
            **path_params (str): Named path parameters (e.g. uuid=session_uuid).

        Returns:
            str: Formatted path (e.g. /da/explain/{uuid}).
        """
        return get_contract(name).path.format(**path_params)

    def _build_request_body(
        self,
        cd_result_reference: Dict[str, str],
        current_observation: Dict[str, float],
        targets: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Optional[List[str]]]] = None,
        feature_penalties: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build request body for explain/enumerate from kwargs.

        Args:
            cd_result_reference (Dict[str, str]): Dict with session_uuid and task_id.
            current_observation (Dict[str, float]): Feature name -> value mapping.
            targets (List[Dict[str, Any]]): Per-target specs: [{"col": "...", "sense": "one of: >, <, in", "threshold": number|[lb, ub]}].  # noqa: E501
            constraints (Optional[Dict[str, Optional[List[str]]]]): Optional constraints.
            feature_penalties (Optional[List[str]]): Optional feature names to penalize.
            params (Optional[Dict[str, Any]]): Optional algorithm params dict.

        Returns:
            Dict[str, Any]: Request body for POST /da/explain or /da/enumerate.
        """
        body: Dict[str, Any] = {
            "cd_result_reference": cd_result_reference,
            "current_observation": current_observation,
            "targets": targets,
        }
        body.update(
            {
                k: v
                for k, v in (
                    ("constraints", constraints),
                    ("feature_penalties", feature_penalties),
                    ("params", params),
                )
                if v is not None
            }
        )
        return body

    async def run_explain(
        self,
        session_uuid: str,
        cd_result_reference: Dict[str, str],
        current_observation: Dict[str, float],
        targets: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Optional[List[str]]]] = None,
        feature_penalties: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a DA explain task (enqueue and return queued response).

        Args:
            session_uuid (str): Session UUID (must match cd_result_reference if needed).
            cd_result_reference (Dict[str, str]): session_uuid and task_id of CD result.
            current_observation (Dict[str, float]): Feature name -> value mapping.
            targets (List[Dict[str, Any]]): Per-target specs: [{"col": "...", "sense": "one of: >, <, in", "threshold": number|[lb, ub]}].  # noqa: E501
            constraints (Optional[Dict[str, Optional[List[str]]]]): Optional feature constraints.
            feature_penalties (Optional[List[str]]): Optional feature names to penalize.
            params (Optional[Dict[str, Any]]): Optional algorithm params (alpha, time_limit, etc.).

        Returns:
            Dict: Queued response with uuid, task_id, status.
        """
        body = self._build_request_body(
            cd_result_reference=cd_result_reference,
            current_observation=current_observation,
            targets=targets,
            constraints=constraints,
            feature_penalties=feature_penalties,
            params=params,
        )
        payload = DATaskRequestContract.model_validate(body)
        path = self._contract_path("da.run_explain", uuid=session_uuid)
        json_data = payload.model_dump(exclude_none=True)
        response = await self._make_request("POST", path, json_data=json_data)
        return response

    async def run_enumerate(
        self,
        session_uuid: str,
        cd_result_reference: Dict[str, str],
        current_observation: Dict[str, float],
        targets: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Optional[List[str]]]] = None,
        feature_penalties: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a DA enumerate task (enqueue and return queued response).

        Same arguments as run_explain; uses POST /da/enumerate/{uuid}.

        Args:
            session_uuid (str): Session UUID.
            cd_result_reference (Dict[str, str]): session_uuid and task_id of CD result.
            current_observation (Dict[str, float]): Feature name -> value mapping.
            targets (List[Dict[str, Any]]): Per-target specs: [{"col": "...", "sense": "one of: >, <, in", "threshold": number|[lb, ub]}].  # noqa: E501
            constraints (Optional[Dict[str, Optional[List[str]]]]): Optional feature constraints.
            feature_penalties (Optional[List[str]]): Optional feature names to penalize.
            params (Optional[Dict[str, Any]]): Optional algorithm params.

        Returns:
            Dict: Queued response with uuid, task_id, status.
        """
        body = self._build_request_body(
            cd_result_reference=cd_result_reference,
            current_observation=current_observation,
            targets=targets,
            constraints=constraints,
            feature_penalties=feature_penalties,
            params=params,
        )
        payload = DATaskRequestContract.model_validate(body)
        path = self._contract_path("da.run_enumerate", uuid=session_uuid)
        json_data = payload.model_dump(exclude_none=True)
        response = await self._make_request("POST", path, json_data=json_data)
        return response

    async def get_task_status(self, session_uuid: str, task_id: str) -> Dict[str, Any]:
        """Get the status of a DA task.

        Args:
            session_uuid (str): Session UUID.
            task_id (str): DA task ID.

        Returns:
            Dict: Status with uuid, task_id, status, optional error.
        """
        path = self._contract_path("da.get_task_status", uuid=session_uuid)
        params = {"task_id": task_id}
        response = await self._make_request("GET", path, params=params)
        return response

    async def get_task_result(self, session_uuid: str, task_id: str) -> Dict[str, Any]:
        """Get the result URL for a completed DA task.

        Args:
            session_uuid (str): Session UUID.
            task_id (str): DA task ID.

        Returns:
            Dict: Result with result_url and expires_in.
        """
        path = self._contract_path("da.get_task_result", uuid=session_uuid)
        params = {"task_id": task_id}
        response = await self._make_request("GET", path, params=params)
        return response

    async def delete_task(self, session_uuid: str, task_id: str) -> None:
        """Delete a DA task and associated S3 artifacts.

        Args:
            session_uuid (str): Session UUID.
            task_id (str): DA task ID.
        """
        path = self._contract_path("da.delete_task", uuid=session_uuid)
        params = {"task_id": task_id}
        await self._make_request("DELETE", path, params=params)

    async def wait_for_task(
        self,
        session_uuid: str,
        task_id: str,
        timeout: int = 300,
        interval: int = 5,
    ) -> Dict[str, Any]:
        """Wait for a DA task to complete (poll get_task_status until succeeded or failed).

        Args:
            session_uuid (str): Session UUID.
            task_id (str): DA task ID.
            timeout (int): Maximum time to wait in seconds (default 300).
            interval (int): Time between polls in seconds (default 5).

        Returns:
            Dict: Final status (succeeded or failed).
        """
        from causal_ai_sdk.exceptions import APIError, ValidationError

        async def check() -> Dict[str, Any]:
            return await self.get_task_status(session_uuid, task_id)

        def on_failed(status: Dict[str, Any]) -> None:
            error_msg = status.get("error") or "Unknown error"
            raise ValidationError(
                message=f"DA task failed: {error_msg}",
                status_code=None,
                response_data=None,
            )

        return await poll_until_ready_or_fail(
            check_func=check,
            is_ready=lambda s: s.get("status") == "succeeded",
            is_failed=lambda s: s.get("status") == "failed",
            on_failed=on_failed,
            timeout=timeout,
            interval=interval,
            timeout_error_message=f"DA task did not complete within {timeout}s",
            retry_exceptions=(APIError,),
            retry_if=lambda e: getattr(e, "status_code", None) in (500, 503),
        )
