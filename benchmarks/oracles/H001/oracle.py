from library.core import canonicalize_email, deduplicate_emails


def test_email_contract():
    assert canonicalize_email("  Alice@Example.COM ") == "alice@example.com"
    assert deduplicate_emails(
        [" Alice@Example.com ", "BOB@example.com", "alice@example.COM"]
    ) == ["alice@example.com", "bob@example.com"]
