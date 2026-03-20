"""SDK contract registry for static OpenAPI compatibility checks."""

from causal_ai_sdk.contracts.endpoints import CONTRACTS
from causal_ai_sdk.contracts.types import EndpointContract

_CONTRACTS_BY_NAME = {contract.name: contract for contract in CONTRACTS}


def get_contract(name: str) -> EndpointContract:
    """Get a contract by unique endpoint name.

    Args:
        name (str): Endpoint contract name, for example ``kg.init_session``.

    Returns:
        EndpointContract: Matched contract definition.

    Raises:
        KeyError: If no contract is registered for ``name``.
    """
    try:
        return _CONTRACTS_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"Unknown contract name: {name}") from exc


__all__ = ["CONTRACTS", "EndpointContract", "get_contract"]
