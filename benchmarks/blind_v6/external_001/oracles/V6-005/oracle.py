import pytest
from strip_balanced_brackets import strip_balanced_brackets

# Reference bracket pairing
BRACKET_MAP = {
    '(': ')',
    '[': ']',
    '{': '}',
    '<': '>',
}

# Reference oracle for expected results (deterministic, no target call):
def ref_strip_balanced_brackets(texts, bracket):
    if bracket not in BRACKET_MAP:
        raise ValueError('Invalid bracket supplied')
    closing = BRACKET_MAP[bracket]
    out = []
    for s in texts:
        if len(s) >= 2 and s[0] == bracket and s[-1] == closing:
            out.append(s[1:-1])
        else:
            out.append(s)
    return out

# --- Tests ---

def test_behavioral_examples():
    # Provided by requirement
    # (1)
    assert strip_balanced_brackets(['(foo)', '(bar'], '(') == ['foo', '(bar']
    # (2)
    assert strip_balanced_brackets(['[abc]', '[abc]]', 'abc]'], '[') == ['abc', '[abc]]', 'abc]']
    # (3)
    assert strip_balanced_brackets(['{x}', '{{x}}'], '{') == ['x', '{x}']
    # (4)
    assert strip_balanced_brackets(['<a>', '<a>>'], '<') == ['a', '<a>>']
    # (5)
    assert strip_balanced_brackets([''], '[') == ['']
    # (6)
    with pytest.raises(ValueError):
        strip_balanced_brackets(['foo'], 'x')


def test_strip_outermost_only_and_unaffected_cases():
    # Input is an empty string
    assert strip_balanced_brackets([''], '(') == ['']
    # Input is a string with just the pair: becomes empty string
    for b in BRACKET_MAP:
        pair = b + BRACKET_MAP[b]
        assert strip_balanced_brackets([pair], b) == ['']
    # Input of length one (cannot possibly match)
    assert strip_balanced_brackets(['('], '(') == ['(']
    assert strip_balanced_brackets(['}'], '{') == ['}']
    # Input with whitespace at boundary (brackets not at boundary, so remains unchanged)
    test_cases = [
        ' (foo)',  # bracket at index 1
        '(bar) ',  # closing at index -2
        ' (baz) ',  # bracket at index 1, closing at index -2
    ]
    assert strip_balanced_brackets(test_cases, '(') == test_cases
    # String with brackets reversed, or only one at a position
    assert strip_balanced_brackets([')foo('], '(') == [')foo(']
    assert strip_balanced_brackets(['(foo]'], '(') == ['(foo]']
    assert strip_balanced_brackets(['foo)'], '(') == ['foo)']
    # Nested: should only remove ONE outermost if both at edges
    assert strip_balanced_brackets(['((bar))'], '(') == ['(bar)']
    assert strip_balanced_brackets(['{{abc}}'], '{') == ['{abc}']


def test_no_input_modification_and_varied_input_types():
    # Check that input list and elements are not mutated (no side effects)
    input_list = ['(abc)', '(def]']
    input_copy = input_list[:]
    element_copy = input_list[0][:], input_list[1][:]
    result = strip_balanced_brackets(input_list, '(')
    # Output is as expected by reference oracle
    assert result == ref_strip_balanced_brackets(input_copy, '(')
    # Inputs are not changed
    assert input_list == input_copy
    # Input element strings unchanged
    assert input_list[0] == element_copy[0]
    assert input_list[1] == element_copy[1]
    # Repeated calls produce identical deterministic results
    r1 = strip_balanced_brackets(['[foo]', '[bar'], '[')
    r2 = strip_balanced_brackets(['[foo]', '[bar'], '[')
    assert r1 == r2 == ['foo', '[bar']


def test_valueerror_for_invalid_bracket_argument():
    # Test all ASCII punctuation except the 4 valid opens
    import string
    invalids = set(string.punctuation) - set(BRACKET_MAP.keys())
    case = ['stuff']
    # Try a mix of invalids (just some to be deterministic)
    for ch in list(sorted(invalids))[:10]:  # fixed 10 for deterministic
        with pytest.raises(ValueError):
            strip_balanced_brackets(case, ch)
    # Unicode bracket
    with pytest.raises(ValueError):
        strip_balanced_brackets(['1foo7'], '1')  # '«' and '»'


def test_examples_with_mixed_legality_and_edges():
    # Some correct, some not:
    mixed = [
        '[edge]',    # to strip
        'edge]',     # no leading open
        '[edge',     # no trailing close
        '[something]',
        'normal',    # no brackets
        '',          # empty
    ]
    expected = ref_strip_balanced_brackets(mixed, '[')
    got = strip_balanced_brackets(mixed, '[')
    assert got == expected
    # Input including an already empty string
    assert strip_balanced_brackets([''], '<') == ['']
    # Brackets as inner text not stripped
    assert strip_balanced_brackets(['foo[bar]'], '[') == ['foo[bar]']
