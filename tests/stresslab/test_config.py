from pathlib import Path
import tomllib

import pytest
from pydantic import ValidationError

from stresslab import __all__
from stresslab.config import RuntimeProfile, load_profile


def test_package_exports_version():
    assert "__version__" in __all__


def test_pyproject_declares_stresslab_packaging():
    data = tomllib.loads(Path("pyproject.toml").read_text())

    assert data["project"]["optional-dependencies"]["stresslab"] == [
        "httpx>=0.27",
        "numpy>=1.26",
        "pdfplumber>=0.11",
    ]
    assert "stresslab" in data["tool"]["hatch"]["build"]["targets"]["wheel"][
        "packages"
    ]


def test_load_lan_profile():
    profile = load_profile("lan")

    assert str(profile.llm_base_url).removesuffix("/") == "http://192.168.100.207:8000"
    assert str(profile.embedding_url).endswith("/v1/embeddings")


def test_load_tunnel_profile():
    profile = load_profile("tunnel")

    assert str(profile.llm_base_url).removesuffix("/") == "http://127.0.0.1:8000"
    assert str(profile.embedding_url).removesuffix("/") == "http://127.0.0.1:8001/v1/embeddings"


def test_load_profile_rejects_unknown_name():
    try:
        load_profile("missing")
    except ValueError as exc:
        assert "Unknown profile" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_profile_rejects_path_traversal_name():
    with pytest.raises(ValueError, match="Invalid profile name"):
        load_profile("../public")


def test_runtime_profile_rejects_non_positive_top_k():
    with pytest.raises(ValidationError):
        RuntimeProfile(
            name="test",
            llm_base_url="http://example.com",
            embedding_url="http://example.com/v1/embeddings",
            answer_model="answer",
            embedding_model="embedding",
            top_k=0,
        )
