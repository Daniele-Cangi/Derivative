import pytest
from normalize_sections import normalize_sections

@pytest.mark.parametrize(
    "input_sections, expected",
    [
        # Behavioral test 1
        (['Introduction', 'Summary  ', 'Appendix-B'], ['INTRODUCTION', 'SUMMARY', 'APPENDIX_B']),
        # Behavioral test 2
        (['  section--3 ', '\tAnnex--A', '--', ''], ['SECTION_3', 'ANNEX_A', '', '']),
        # Behavioral test 3
        (['  ', 'TABLE 1-2-3'], ['', 'TABLE_1_2_3']),
    ]
)
def test_normalize_sections_behavioral(input_sections, expected):
    result = normalize_sections(input_sections)
    assert result == expected

def test_normalize_sections_various_separators():
    # input contains mixed whitespace, tabs and dashes, some at start/end
    input_sections = [
        '---Foo	Bar',   # Many dashes at start: dash/tab to underscore, upper
        'baz - qux',     # Space/dash/space
        '   ',           # all-whitespace, becomes ''
        '\t- - -',       # tabs and dashes, becomes ''
        'alpha--- beta--gamma ' # in-between/at end
    ]
    expected = [
        'FOO_BAR',
        'BAZ_QUX',
        '',
        '',
        'ALPHA_BETA_GAMMA'
    ]
    result = normalize_sections(input_sections)
    assert result == expected

def test_normalize_sections_preserves_length_and_order():
    input_sections = ['first', '  SECOND   -third', '', ' 	 ']
    # Confirm length preserved and correct normalization
    expected = ['FIRST', 'SECOND_THIRD', '', '']
    result = normalize_sections(input_sections)
    assert len(result) == len(input_sections)
    assert result == expected

def test_normalize_sections_edge_cases():
    # Only dashes/whitespace/tab, or empty/just one char
    input_sections = ['-', '--', '', ' ', '\t', 'a', 'A--', 'B   ', '--C--']
    expected = ['', '', '', '', '', 'A', 'A', 'B', 'C']
    result = normalize_sections(input_sections)
    assert result == expected

def test_normalize_sections_idempotent():
    # normalization should be idempotent
    src = ['This--is  a	Test', '', '   	', 'A--B']
    once = normalize_sections(src)
    twice = normalize_sections(once)
    assert twice == once
