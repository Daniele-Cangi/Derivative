import os
import shutil
import pytest

from your_module import rename_directory_tree

@pytest.fixture
def setup_temp_dir(tmp_path):
    # Create a directory tree with files and symlinks
    src = tmp_path / "src"
    src.mkdir()
    (src / "file1.txt").write_text("content1")
    subdir = src / "subdir"
    subdir.mkdir()
    (subdir / "file2.txt").write_text("content2")
    # Create a symlink inside src pointing to file1.txt
    os.symlink(src / "file1.txt", src / "link_to_file1")
    # Create a symlink inside subdir pointing to ../file1.txt
    os.symlink(os.path.join("..", "file1.txt"), subdir / "link_to_file1_rel")
    return src

@pytest.fixture
def target_dir(tmp_path):
    return tmp_path / "dest"


def test_successful_rename_preserves_content_and_symlinks(setup_temp_dir, target_dir):
    src = str(setup_temp_dir)
    dest = str(target_dir)
    rename_directory_tree(src, dest)
    # src should no longer exist
    assert not os.path.exists(src)
    # dest should exist
    assert os.path.isdir(dest)
    # Check files preserved
    assert os.path.isfile(os.path.join(dest, "file1.txt"))
    assert open(os.path.join(dest, "file1.txt"), "r").read() == "content1"
    assert os.path.isfile(os.path.join(dest, "subdir", "file2.txt"))
    assert open(os.path.join(dest, "subdir", "file2.txt"), "r").read() == "content2"
    # Check symlinks preserved and point to correct targets
    link_path = os.path.join(dest, "link_to_file1")
    assert os.path.islink(link_path)
    target = os.readlink(link_path)
    assert target.endswith("file1.txt")
    sublink_path = os.path.join(dest, "subdir", "link_to_file1_rel")
    assert os.path.islink(sublink_path)
    subtarget = os.readlink(sublink_path)
    assert subtarget == os.path.join("..", "file1.txt")


def test_rollback_on_partial_failure(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "file.txt").write_text("data")
    dest = tmp_path / "dest"
    dest.mkdir()
    # We monkeypatch os.rename to fail halfway during renaming
    original_rename = os.rename
    call_count = {"count": 0}
    
    def failing_rename(src_path, dst_path):
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise OSError("Dummy failure during rename")
        return original_rename(src_path, dst_path)

    os.rename = failing_rename
    try:
        with pytest.raises(OSError, match="Dummy failure"):
            rename_directory_tree(str(src), str(dest / "moved"))
        # Source dir should remain intact
        assert os.path.isdir(str(src))
        assert os.path.isfile(str(src / "file.txt"))
        # Destination should not have partial data
        assert not os.path.exists(str(dest / "moved"))
    finally:
        os.rename = original_rename


def test_raises_oserror_if_source_not_exist(tmp_path):
    src = tmp_path / "nonexistent"
    dest = tmp_path / "dest"
    with pytest.raises(OSError):
        rename_directory_tree(str(src), str(dest))


def test_raises_oserror_if_dest_already_exists(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(OSError):
        rename_directory_tree(str(src), str(dest))


def test_preserves_file_permissions(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    file = src / "file_perm.txt"
    file.write_text("hello")
    # Set restrictive permission
    original_mode = 0o640
    os.chmod(file, original_mode)
    dest = tmp_path / "dest"
    rename_directory_tree(str(src), str(dest))
    moved_file = dest / "file_perm.txt"
    assert moved_file.exists()
    moved_mode = os.stat(moved_file).st_mode & 0o777
    assert moved_mode == original_mode
