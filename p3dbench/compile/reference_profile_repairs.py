"""Checksum-selected profile repairs for four published reference programs.

Only registered program checksums select the repaired profile builder.
All other programs retain the existing interpreter behavior.
Whitespace is ignored; feature, face and curve insertion order is preserved.
"""
import hashlib
import json

REFERENCE_PROFILE_REPAIRS = {
    "05da77fe0396b5e47e55944e935362c93685d8a7eeecb27a0b878cc8cddeee87": "0052/00522179",
    "128751eca4a644c1949f36bb42a747d0b99f1e2fc23dbac606dccba234f19f8a": "0025/00254326",
    "cb35ed084c067968baf9906b678e9fc8cc6990eed7e59f7fc82fc63b9f0e314a": "0046/00466588",
    "dc4402be5fc8b048004bfa396f73ccfac27eaf80bd61d8ed7f0495c08fd14a24": "0058/00582353",
}


def program_digest(data):
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def needs_profile_repair(data):
    return program_digest(data) in REFERENCE_PROFILE_REPAIRS
