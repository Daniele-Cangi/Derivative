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
            "observed": ["error: invalid input\n"],
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
