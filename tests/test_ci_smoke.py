import pathlib


def test_chat_curl_uses_failfast_flag() -> None:
    script_path = pathlib.Path('tools/ci/smoke.sh')
    contents = script_path.read_text().splitlines()

    curl_lines = [line for line in contents if line.strip().startswith('curl')]
    assert len(curl_lines) >= 2, 'Expected at least two curl commands in smoke.sh'

    second_curl = curl_lines[1]
    assert any(
        token.startswith('-') and 'f' in token[1:]
        for token in second_curl.split()
    ), 'Second curl invocation must include -f flag'


def test_all_curl_commands_use_failfast_flag() -> None:
    script_path = pathlib.Path('tools/ci/smoke.sh')
    contents = script_path.read_text().splitlines()

    curl_lines = [line.strip() for line in contents if line.strip().startswith('curl')]
    assert curl_lines, 'Expected at least one curl command in smoke.sh'

    for idx, curl_line in enumerate(curl_lines, start=1):
        tokens = curl_line.split()
        assert any(
            token.startswith('-') and 'f' in token.lstrip('-')
            for token in tokens[1:]
        ), f'curl command #{idx} must include the -f flag'
