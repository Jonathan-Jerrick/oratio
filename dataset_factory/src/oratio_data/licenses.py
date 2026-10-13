"""License classes and the rule for what may be published in OCDD."""

from enum import StrEnum


class LicenseClass(StrEnum):
    PUBLIC_DOMAIN = "public_domain"
    CC0 = "cc0"
    CC_BY = "cc_by"
    CC_BY_SA = "cc_by_sa"
    OWN = "own"
    APACHE = "apache"
    MIXED_CC = "mixed_cc"
    NC_RESEARCH = "nc_research"


PUBLISHABLE = frozenset(
    {
        LicenseClass.PUBLIC_DOMAIN,
        LicenseClass.CC0,
        LicenseClass.CC_BY,
        LicenseClass.CC_BY_SA,
        LicenseClass.OWN,
    }
)


class LicenseError(ValueError):
    """A clip's license does not allow it into the published dataset."""


def require_publishable(license_class: str, item: str) -> LicenseClass:
    """Return the parsed class, or refuse the item if it cannot be redistributed."""
    try:
        parsed = LicenseClass(license_class)
    except ValueError as error:
        raise LicenseError(f"{item}: unknown license class {license_class!r}") from error
    if parsed not in PUBLISHABLE:
        raise LicenseError(f"{item}: {parsed.value} material cannot be published in OCDD")
    return parsed
