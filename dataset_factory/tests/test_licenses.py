import pytest

from oratio_data.licenses import PUBLISHABLE, LicenseClass, LicenseError, require_publishable


@pytest.mark.parametrize("name", ["public_domain", "cc0", "cc_by", "cc_by_sa", "own"])
def test_redistributable_classes_are_publishable(name: str) -> None:
    assert require_publishable(name, "clip") is LicenseClass(name)


@pytest.mark.parametrize("name", ["nc_research", "mixed_cc", "apache"])
def test_benchmark_only_classes_are_refused(name: str) -> None:
    with pytest.raises(LicenseError, match="cannot be published"):
        require_publishable(name, "clip-7")


def test_unknown_class_is_refused_by_name() -> None:
    with pytest.raises(LicenseError, match="clip-7: unknown license class 'nd_private'"):
        require_publishable("nd_private", "clip-7")


def test_whitelist_is_exactly_the_planned_five() -> None:
    assert {c.value for c in PUBLISHABLE} == {"public_domain", "cc0", "cc_by", "cc_by_sa", "own"}
