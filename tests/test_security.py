"""Unit tests for SecurityEnforcer and SecurityPolicyConfig."""

import pytest

from authentipix.metadata.context import AnalysisContext
from authentipix.metadata.security import (
    SecurityEnforcer,
    SecurityPolicyConfig,
    SecurityViolationError,
)


def test_security_policy_file_size_breach():
    policy = SecurityPolicyConfig(max_file_size_bytes=100)
    enforcer = SecurityEnforcer(policy)

    raw_bytes = b"X" * 200
    ctx = AnalysisContext.from_bytes(raw_bytes, mime_type="image/jpeg")

    with pytest.raises(SecurityViolationError) as exc_info:
        enforcer.validate_context(ctx)

    assert "exceeds maximum policy limit" in str(exc_info.value)


def test_security_policy_raster_dimension_breach():
    policy = SecurityPolicyConfig(max_pixel_count=1000)  # Max 1,000 pixels
    enforcer = SecurityEnforcer(policy)

    # 100 x 100 = 10,000 pixels > 1,000 limit
    with pytest.raises(SecurityViolationError) as exc_info:
        enforcer.validate_raster_dimensions(100, 100)

    assert "exceeds maximum policy limit" in str(exc_info.value)


def test_safe_parse_xml_valid():
    valid_xml = "<xmp><title>Sample</title></xmp>"
    elem = SecurityEnforcer.safe_parse_xml(valid_xml)
    assert elem is not None
    assert elem.tag == "xmp"


def test_safe_parse_xml_xxe_entity_expansion_blocked():
    # XXE entity expansion attempt payload
    xxe_xml = """<?xml version="1.0" encoding="ISO-8859-1"?>
    <!DOCTYPE foo [
    <!ELEMENT foo ANY >
    <!ENTITY xxe SYSTEM "file:///etc/passwd" >]>
    <foo>&xxe;</foo>"""

    with pytest.raises(SecurityViolationError):
        SecurityEnforcer.safe_parse_xml(xxe_xml)
