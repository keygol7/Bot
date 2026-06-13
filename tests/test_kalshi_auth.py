import base64

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from bot.config.settings import KalshiConfig
from bot.venues.kalshi import KalshiVenue, build_signature_headers


def test_signature_verifies_with_public_key():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    headers = build_signature_headers(
        "kid123", key, "GET", "/trade-api/v2/markets", timestamp_ms="1700000000000"
    )
    assert headers["KALSHI-ACCESS-KEY"] == "kid123"
    assert headers["KALSHI-ACCESS-TIMESTAMP"] == "1700000000000"

    message = "1700000000000" + "GET" + "/trade-api/v2/markets"
    sig = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    # Raises InvalidSignature if the signature is wrong — passes silently if valid.
    key.public_key().verify(
        sig, message.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_unauthenticated_when_no_credentials():
    v = KalshiVenue(KalshiConfig(api_key_id="", private_key_path=""))
    assert v.authenticated is False
    assert v._auth_headers("GET", "/markets") == {}  # no signing without creds


def test_authenticated_flag_with_credentials():
    v = KalshiVenue(KalshiConfig(api_key_id="kid", private_key_path="/tmp/key.pem"))
    assert v.authenticated is True


def test_base_path_extracted_from_api_base():
    v = KalshiVenue(KalshiConfig(api_base="https://api.elections.kalshi.com/trade-api/v2"))
    assert v._base_path == "/trade-api/v2"
