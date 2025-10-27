from src.orch import server as orch_server


def test_make_error_body_sanitizes_server_errors() -> None:
    body = orch_server._make_error_body(
        status_code=500,
        message="ValueError: secret detail",
        error_type="provider_server_error",
    )
    assert body["error"]["message"] == "internal server error"


def test_make_error_body_preserves_client_errors() -> None:
    message = "invalid request payload"
    body = orch_server._make_error_body(
        status_code=400,
        message=message,
        error_type="provider_error",
    )
    assert body["error"]["message"] == message


def test_make_error_body_strips_stack_trace_from_client_errors() -> None:
    body = orch_server._make_error_body(
        status_code=400,
        message="Traceback (most recent call last):\nValueError: bad",
        error_type="provider_error",
    )
    assert body["error"]["message"] == "Traceback (most recent call last):"
