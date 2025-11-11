from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

MODULE_PATH = Path(__file__).resolve().parents[3] / "tools" / "download_assets.py"
SPEC = importlib.util.spec_from_file_location("download_assets", MODULE_PATH)
download_assets = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(download_assets)  # type: ignore[assignment]


def test_download_invokes_snapshot(monkeypatch, tmp_path: Path) -> None:
    calls = {}

    def fake_snapshot_download(*, repo_id, local_dir, local_dir_use_symlinks, revision, token, resume_download):
        calls["repo_id"] = repo_id
        calls["local_dir"] = local_dir
        calls["revision"] = revision
        calls["token"] = token
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return local_dir

    monkeypatch.setattr(download_assets, "snapshot_download", fake_snapshot_download)

    target = tmp_path / "model"
    result = download_assets.download("org/repo", target, revision="main", token="secret")
    assert Path(result).exists()
    assert calls == {
        "repo_id": "org/repo",
        "local_dir": str(target),
        "revision": "main",
        "token": "secret",
    }


def test_main_uses_env_token(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_download(repo_id, target, revision, token):
        captured["repo_id"] = repo_id
        captured["token"] = token
        return tmp_path

    monkeypatch.setattr(download_assets, "download", fake_download)
    monkeypatch.setenv("HF_TOKEN", "env-secret")

    exit_code = download_assets.main(["--model-id", "org/repo"])
    assert exit_code == 0
    assert captured == {"repo_id": "org/repo", "token": "env-secret"}
