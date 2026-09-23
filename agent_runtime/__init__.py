"""agent-runtime: a durable, checkpointed state-machine executor for LLM agents."""

from .checkpoint import Checkpoint, Checkpointer, MemoryCheckpointer, SQLiteCheckpointer
from .graph import CycleWarning, Graph, GraphReport, GraphValidationError, RoutingError
from .guards import TenantMismatchError, sanitize_input, strip_control_chars, tenant_filter
from .models import (
    AllTiersFailedError,
    DeterministicFallback,
    FallbackChain,
    MockModel,
    ModelClient,
    ModelError,
    ModelResponse,
    ModelTimeoutError,
    TierAttempt,
)
from .runtime import (
    RESOLUTION_MARKER,
    ResolutionHook,
    RunFailedError,
    Runtime,
    UnknownRunError,
    find_consecutive_repeat,
)
from .state import AgentState, Output, RunError, Step
from .structured import DegradedResult, JSONExtractionError, SchemaAttempt, enforce_schema
from .telemetry import OpenTelemetryTracer, Span, Tracer, current_tracer, get_tracer
from .tools import (
    Tool,
    ToolError,
    ToolInputError,
    ToolNotAllowedError,
    ToolNotFoundError,
    ToolOutputError,
    ToolRegistry,
    ToolTimeoutError,
    default_registry,
    tool,
)

__version__ = "0.1.0"

__all__ = [
    "RESOLUTION_MARKER",
    "AgentState",
    "AllTiersFailedError",
    "Checkpoint",
    "Checkpointer",
    "CycleWarning",
    "DegradedResult",
    "DeterministicFallback",
    "FallbackChain",
    "Graph",
    "GraphReport",
    "GraphValidationError",
    "JSONExtractionError",
    "MemoryCheckpointer",
    "MockModel",
    "ModelClient",
    "ModelError",
    "ModelResponse",
    "ModelTimeoutError",
    "OpenTelemetryTracer",
    "Output",
    "ResolutionHook",
    "RoutingError",
    "RunError",
    "RunFailedError",
    "Runtime",
    "SQLiteCheckpointer",
    "SchemaAttempt",
    "Span",
    "Step",
    "TenantMismatchError",
    "TierAttempt",
    "Tool",
    "ToolError",
    "ToolInputError",
    "ToolNotAllowedError",
    "ToolNotFoundError",
    "ToolOutputError",
    "ToolRegistry",
    "ToolTimeoutError",
    "Tracer",
    "UnknownRunError",
    "__version__",
    "current_tracer",
    "default_registry",
    "enforce_schema",
    "find_consecutive_repeat",
    "get_tracer",
    "sanitize_input",
    "strip_control_chars",
    "tenant_filter",
    "tool",
]
