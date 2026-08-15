import pytest
import unicodedata
import io
from collections import Counter
from grapheme_frequency import count_grapheme_frequencies

@pytest.fixture
def small_text_file(tmp_path):
    file_path = tmp_path / "small.txt"
    content = "aááäb600b"
    # This string contains 'a', 'á' x2, 'ä', 'b' x2, and 😀 (grinning face emoji)
    file_path.write_text(content, encoding='utf-8')
    return file_path

@pytest.fixture
def empty_text_file(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding='utf-8')
    return file_path

@pytest.fixture
def large_text_file(tmp_path):
    file_path = tmp_path / "large.txt"
    # write 1_000 repetitions of 'abc', 1_000 of '😀', and 999 of 'á' to simulate large file
    # As size should not exceed available memory, keep moderate size (few MBs)
    part1 = 'abc' * 1000  # 3,000 chars
    part2 = '600' * 1000  # 1,000 emojis
    part3 = 'á' * 999
    content = part1 + part2 + part3
    file_path.write_text(content, encoding='utf-8')
    return file_path

@pytest.mark.parametrize("input_text,expected_sorted", [
    ("aááäb600b", [('á', 2), ('b', 2), ('a', 1), ('ä', 1), ('600', 1)]),
    ("", []),
    ("eeeèéêëe", [('e', 4), ('è', 1), ('é', 1), ('ê', 1), ('ë', 1)]),
])
def test_count_grapheme_frequencies_correctness(tmp_path, input_text, expected_sorted):
    file_path = tmp_path / "testfile.txt"
    file_path.write_text(input_text, encoding='utf-8')
    result = count_grapheme_frequencies(str(file_path))
    assert isinstance(result, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in result)
    # Check sorting: frequency descending, then lex ascending
    sorted_expected = sorted(expected_sorted, key=lambda t: (-t[1], t[0]))
    assert result == sorted_expected


def test_count_grapheme_frequencies_small_file(small_text_file):
    result = count_grapheme_frequencies(str(small_text_file))
    expected = [('á', 2), ('b', 2), ('a', 1), ('ä', 1), ('600', 1)]
    assert result == expected


def test_count_grapheme_frequencies_empty_file(empty_text_file):
    result = count_grapheme_frequencies(str(empty_text_file))
    assert result == []


def test_count_grapheme_frequencies_large_file(large_text_file):
    # We expect 'a':1000 count (from 'abc'*1000), 'b':1000, 'c':1000, 
    # '😀':1000, and 'á':999
    result = count_grapheme_frequencies(str(large_text_file))
    expected_counts = {'a': 1000, 'b': 1000, 'c': 1000, '600': 1000, 'á': 999}
    counts_dict = dict(result)
    assert counts_dict == expected_counts
    # Now check sorting: frequencies descending, ties lex ascending
    freqs = list(counts_dict.values())
    assert freqs == sorted(freqs, reverse=True)
    # For items with same frequency 1000, order lex asc
    same_freq_keys = [k for k,v in result if v == 1000]
    assert same_freq_keys == sorted(same_freq_keys)


def test_count_grapheme_frequencies_invalid_path():
    with pytest.raises(FileNotFoundError):
        count_grapheme_frequencies("non_existent_file.txt")
