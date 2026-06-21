import base64

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)
from cryptography.exceptions import InvalidSignature

from bot.venues.polymarket_us import build_auth_headers, load_ed25519_key


def test_load_key_from_raw_base64_secret():
    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    secret_b64 = base64.b64encode(seed).decode()

    loaded = load_ed25519_key(secret_key=secret_b64)
    # Same key -> same public key bytes.
    assert loaded.public_key().public_bytes_raw() == key.public_key().public_bytes_raw()


def test_load_key_from_pem(tmp_path):
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    p = tmp_path / "qcex.pem"
    p.write_bytes(pem)

    loaded = load_ed25519_key(pem_path=str(p))
    assert loaded.public_key().public_bytes_raw() == key.public_key().public_bytes_raw()


def test_auth_headers_signature_verifies():
    key = Ed25519PrivateKey.generate()
    headers = build_auth_headers(
        "mykeyid", key, "GET", "/v1/portfolio/positions", timestamp_ms="1700000000000"
    )
    assert headers["X-PM-Access-Key"] == "mykeyid"
    assert headers["X-PM-Timestamp"] == "1700000000000"

    message = "1700000000000" + "GET" + "/v1/portfolio/positions"
    sig = base64.b64decode(headers["X-PM-Signature"])
    key.public_key().verify(sig, message.encode())  # raises InvalidSignature if wrong


def test_auth_headers_signature_rejects_tampering():
    key = Ed25519PrivateKey.generate()
    headers = build_auth_headers("kid", key, "GET", "/v1/orders", timestamp_ms="1")
    sig = base64.b64decode(headers["X-PM-Signature"])
    with pytest.raises(InvalidSignature):
        key.public_key().verify(sig, b"1GET/v1/DIFFERENT")
