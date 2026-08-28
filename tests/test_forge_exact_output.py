from core.forge.exact_output import exact_output_contract_evidence


REQUIREMENT = (
    "If input is invalid, the tool outputs exactly 'error: invalid input' "
    "to stderr and exits with code 1."
)


def test_exact_output_contract_rejects_added_newline():
    evidence = exact_output_contract_evidence(
        REQUIREMENT,
        {
            "src/tool.py": (
                "import sys\n"
                "def fail():\n"
                "    sys.stderr.write('error: invalid input\\n')\n"
            )
        },
    )

    assert evidence == [
        {
            "stream": "stderr",
            "expected": "error: invalid input",
            "source_fragment": "outputs exactly 'error: invalid input' to stderr",
            "precondition": "input is invalid",
            "observed": ["error: invalid input\n"],
            "unbound_observed": [],
            "paths": ["src/tool.py"],
            "passed": False,
            "failure_reason": "exact_output_mismatch",
        }
    ]


def test_exact_output_contract_accepts_literal_without_newline():
    evidence = exact_output_contract_evidence(
        REQUIREMENT,
        {
            "src/tool.py": (
                "import sys\n"
                "def fail():\n"
                "    sys.stderr.write('error: invalid input')\n"
            )
        },
    )

    assert evidence[0]["passed"] is True


def test_print_default_newline_is_observed_exactly():
    evidence = exact_output_contract_evidence(
        REQUIREMENT,
        {
            "src/tool.py": (
                "import sys\n"
                "def fail():\n"
                "    print('error: invalid input', file=sys.stderr)\n"
            )
        },
    )

    assert evidence[0]["observed"] == ["error: invalid input\n"]
    assert evidence[0]["passed"] is False


def test_exact_output_contract_ignores_matching_write_in_test_file():
    evidence = exact_output_contract_evidence(
        REQUIREMENT,
        {
            "src/tool.py": (
                "import sys\n"
                "def fail():\n"
                "    sys.stderr.write('wrong')\n"
            ),
            "tests/test_tool.py": (
                "import sys\n"
                "def test_output():\n"
                "    sys.stderr.write('error: invalid input')\n"
            ),
        },
    )

    assert evidence[0]["passed"] is False
    assert evidence[0]["observed"] == ["wrong"]
    assert evidence[0]["paths"] == ["src/tool.py"]


def test_exact_output_contract_rejects_unrelated_matching_helper():
    evidence = exact_output_contract_evidence(
        REQUIREMENT,
        {
            "src/tool.py": (
                "import sys\n"
                "def fail():\n"
                "    sys.stderr.write('error: invalid input')\n"
                "def main(value):\n"
                "    if value == 'invalid':\n"
                "        sys.stderr.write('wrong')\n"
            )
        },
        target_names={"main"},
    )

    assert evidence[0]["passed"] is False
    assert evidence[0]["observed"] == ["wrong"]
    assert evidence[0]["unbound_observed"] == ["error: invalid input"]


def test_exact_output_contract_accepts_helper_called_from_matching_target_branch():
    evidence = exact_output_contract_evidence(
        REQUIREMENT,
        {
            "src/tool.py": (
                "import sys\n"
                "def fail():\n"
                "    sys.stderr.write('error: invalid input')\n"
                "def main(value):\n"
                "    if value == 'invalid':\n"
                "        fail()\n"
            )
        },
        target_names={"main"},
    )

    assert evidence[0]["passed"] is True


def test_exact_output_contract_accepts_value_error_branch():
    evidence = exact_output_contract_evidence(
        REQUIREMENT,
        {
            "src/tool.py": (
                "import sys\n"
                "def main(value):\n"
                "    try:\n"
                "        int(value)\n"
                "    except ValueError:\n"
                "        sys.stderr.write('error: invalid input')\n"
            )
        },
        target_names={"main"},
    )

    assert evidence[0]["passed"] is True


def test_exact_output_contract_does_not_decode_bytes_literals():
    evidence = exact_output_contract_evidence(
        REQUIREMENT,
        {
            "src/tool.py": (
                "import sys\n"
                "def fail():\n"
                "    sys.stderr.write(b'error: invalid input')\n"
            )
        },
    )

    assert evidence[0]["passed"] is False
    assert evidence[0]["observed"] == []
