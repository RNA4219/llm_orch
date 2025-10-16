import pathlib


def test_all_smoke_curl_commands_use_failfast_flag() -> None:
    script_path = pathlib.Path("tools/ci/smoke.sh")
    contents = script_path.read_text().splitlines()

    curl_lines = [line.strip() for line in contents if line.strip().startswith("curl")]
    assert len(curl_lines) >= 2, "Expected at least two curl commands in smoke.sh"

    first_curl_tokens = curl_lines[0].split()
    assert "-f" in first_curl_tokens, "Health check curl must include -f flag"

    second_curl_tokens = curl_lines[1].split()
    assert "-fs" in second_curl_tokens, "Chat curl must include -fs flags"
