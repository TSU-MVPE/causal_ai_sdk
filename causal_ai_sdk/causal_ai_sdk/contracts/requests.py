"""SDK-side request contract models (static contract layer)."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, StrictStr, model_validator


class KGUploadURLRequestContract(BaseModel):
    """Contract for POST /kg/upload-url/{uuid} request body."""

    filename: Optional[str] = None
    expires_in: Optional[int] = Field(default=900, ge=1)


class KGAddRequestContract(BaseModel):
    """Contract for POST /kg/add/{uuid} request body."""

    title: str
    columns: List[str]
    s3_key: str
    row_count: Optional[int] = Field(default=None, ge=0)
    size_bytes: Optional[int] = Field(default=None, ge=0)


class CDMulticaMatchStartRequestContract(BaseModel):
    """Contract for POST /cd/multica/match/{uuid} request body."""

    s3_key: str
    metadata: Optional[List[Dict]] = None
    params: Optional[Dict] = None


class CDTrackrMatchStartRequestContract(BaseModel):
    """Contract for POST /cd/trackr/match/{uuid} request body."""

    target_s3_key: str
    source_kg_id: str
    target_metadata: Optional[Dict[str, str]] = None
    source_metadata: Optional[Dict[str, str]] = None
    params: Optional[Dict] = None


class CDMatchSetRequestContract(BaseModel):
    """Contract for POST /cd/*/match/{uuid}/set request body."""

    matching: Dict[str, str]
    matching_task_id: Optional[str] = None


class CDMulticaRunRequestContract(BaseModel):
    """Contract for POST /cd/multica/run/{uuid} request body."""

    task_id: str
    s3_key: str
    threshold: Optional[float] = Field(default=None, gt=0, le=1)
    roots: Optional[List[str]] = None
    sinks: Optional[List[str]] = None
    params: Optional[Dict[str, Any]] = None
    matching_task_id: str


class CDLingamRunRequestContract(BaseModel):
    """Contract for POST /cd/lingam/run/{uuid} request body."""

    task_id: StrictStr
    s3_key: StrictStr
    threshold: Optional[float] = Field(default=None, gt=0, le=1)
    params: Optional[Dict[StrictStr, Any]] = None


class CDTrackrRunRequestContract(BaseModel):
    """Contract for POST /cd/trackr/run/{uuid} request body."""

    task_id: str
    s3_key: str
    transferred_knowledge: Dict[str, str]
    threshold: Optional[float] = Field(default=None, gt=0, le=1)
    params: Optional[Dict] = None
    matching_task_id: str

    @model_validator(mode="after")
    def _validate_transferred_knowledge_keys(self) -> "CDTrackrRunRequestContract":
        """Require both session_uuid and kg_id keys in transferred_knowledge.

        Returns:
            CDTrackrRunRequestContract: The validated contract instance.

        Raises:
            ValueError: If either session_uuid or kg_id is missing from
                transferred_knowledge.
        """
        required = {"session_uuid", "kg_id"}
        missing = required - set(self.transferred_knowledge.keys())
        if missing:
            missing_sorted = sorted(missing)
            raise ValueError(
                "transferred_knowledge must include both session_uuid and kg_id; "
                f"missing: {missing_sorted}"
            )
        return self


class DAParamsContract(BaseModel):
    """Optional algorithm params for DA task."""

    model_config = {"extra": "forbid"}

    alpha: Optional[float] = None
    lagrange_multiplier: Optional[float] = None
    constrain_effects_in_bounds: Optional[bool] = None
    time_limit: Optional[float] = None
    verbose: Optional[bool] = None
    n_actions: Optional[int] = None
    key: Optional[StrictStr] = None


class DAConstraintsContract(BaseModel):
    """Optional feature constraints for DA task."""

    model_config = {"extra": "forbid"}

    immutable: Optional[List[StrictStr]] = None
    increase_only: Optional[List[StrictStr]] = None
    decrease_only: Optional[List[StrictStr]] = None
    immutable_effect: Optional[List[StrictStr]] = None


class CDResultReferenceContract(BaseModel):
    """Contract for CD result reference (session_uuid + task_id)."""

    model_config = {"extra": "forbid"}

    session_uuid: StrictStr = Field(..., min_length=1)
    task_id: StrictStr = Field(..., min_length=1)


class DATaskRequestContract(BaseModel):
    """Contract for POST /da/explain/{uuid} and POST /da/enumerate/{uuid} request body."""

    model_config = {"extra": "forbid"}

    cd_result_reference: CDResultReferenceContract
    current_observation: Dict[StrictStr, float] = Field(..., min_length=1)
    targets: List[Dict[StrictStr, Any]] = Field(..., min_length=1)
    constraints: Optional[DAConstraintsContract] = None
    feature_penalties: Optional[List[StrictStr]] = None
    params: Optional[DAParamsContract] = None

    @model_validator(mode="after")
    def _current_observation_may_omit_keys(self) -> "DATaskRequestContract":
        """current_observation may omit keys; omitted keys are imputed at runtime.

        Missing keys are imputed: 0 for one-hot features, column mean for non-one-hot.

        Returns:
            DATaskRequestContract: Self (for chaining).
        """
        return self

    @model_validator(mode="after")
    def _validate_targets_shape_and_thresholds(self) -> "DATaskRequestContract":
        """Validate per-target shape: targets=[{col,sense,threshold}, ...].

        Returns:
            DATaskRequestContract: Self (for chaining).

        Raises:
            ValueError: If targets is empty, has more than one item, or any
                target object has invalid col/sense/threshold shape or values.
        """
        if not self.targets:
            raise ValueError("targets must not be empty")
        if len(self.targets) != 1:
            raise ValueError("multi-target is not enabled yet; targets length must be 1")
        for target in self.targets:
            col = target.get("col")
            sense = target.get("sense")
            threshold = target.get("threshold")
            if not isinstance(col, str) or not col:
                raise ValueError("target object must include non-empty string field 'col'")
            if sense not in (">", "<", "in"):
                raise ValueError("target object field 'sense' must be one of '>', '<', 'in'")
            if threshold is None:
                raise ValueError("target object must include non-null field 'threshold'")
            if sense == "in":
                if not (isinstance(threshold, (list, tuple)) and len(threshold) == 2):
                    raise ValueError(
                        "for target sense 'in', threshold must be [lb, ub] or (lb, ub)"
                    )
                lb, ub = float(threshold[0]), float(threshold[1])
                if lb > ub:
                    raise ValueError(
                        f"threshold tuple must have lower bound <= upper bound, got ({lb}, {ub})"
                    )
            else:
                if isinstance(threshold, (list, tuple)):
                    raise ValueError(
                        "for target sense '>' or '<', threshold must be a numeric value"
                    )
                float(threshold)
        return self
