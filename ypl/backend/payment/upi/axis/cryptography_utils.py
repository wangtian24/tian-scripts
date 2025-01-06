import base64
import hashlib
import os
from typing import Any

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def calculate_request_body_checksum(request_body: dict[str, Any]) -> str:
    """Generate checksum from request map"""
    try:
        concatenated_string = _sanitize_and_concatenate_values(request_body)
        return _hash(concatenated_string.strip())
    except Exception as e:
        print(f"Error generating checksum: {str(e)}")
        raise


def _sanitize(value: str) -> str:
    return value if value and value != "null" else ""


def _sanitize_and_concatenate_values(value: Any) -> str:
    """Recursively process any value and return a concatenated string"""
    if isinstance(value, dict):
        return "".join(_sanitize_and_concatenate_values(v) for k, v in value.items() if k != "checksum")
    elif isinstance(value, list):
        return "".join(_sanitize_and_concatenate_values(item) for item in value)
    else:
        return _sanitize(str(value))


def _hash(data: str) -> str:
    """Generate MD5 hash of input string"""
    try:
        md5_hash = hashlib.md5(data.encode("utf-8"))
        return md5_hash.hexdigest()
    except Exception as e:
        raise RuntimeError("Internal server error") from e


def aes128_encrypt(key: str, plain_text: str) -> str:
    key_bytes = bytes.fromhex(key)

    iv = os.urandom(16)

    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(plain_text.encode("utf-8")) + padder.finalize()

    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv))
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    encrypted_with_iv = iv + encrypted_data

    return base64.b64encode(encrypted_with_iv).decode("utf-8")


def aes128_decrypt(key: str, encrypted_text: str) -> str:
    key_bytes = bytes.fromhex(key)

    encrypted_bytes = base64.b64decode(encrypted_text)

    iv = encrypted_bytes[:16]
    ciphertext = encrypted_bytes[16:]

    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(ciphertext) + decryptor.finalize()

    unpadder = padding.PKCS7(128).unpadder()
    decrypted_data = unpadder.update(padded_data) + unpadder.finalize()

    return decrypted_data.decode("utf-8")
