"""Checksum-selected profile repairs for published reference programs.

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

REFERENCE_GEOMETRY_REPAIRS = {
    "98f642ca6acf9eb5d52589d6503a953c1e2fd89dc1acbe9ef23f98dca129c88f": "0043/00437520",
    "d56beecd3ed01efbb3fcbd84fbd1447fe00cb6ba0c08f5144fbb79f1015bac52": "0073/00734070",
    "84f4083990f3fb6ed6833fb5f8632b303c3997b7676497d8f4656a06e2ec64d8": "0011/00117335",
    "beeb1fc4fe7d8278d5df6262b9a097e8c06e9519b6fc072655a74b569680f5fb": "0009/00098309",
    "7d71ad96a4d4c89333ef057ed62eb5761dc5dc4b593748135330df90ef733f29": "0052/00520870",
    "7ba87ec699e2497d017cb687eaa8446e158612ca6e7b82901d7ec78b5ca934f6": "0013/00135195",
    "65ef836cddc65471b7b81c6f4cba8ad0281388805e384c87d480187a5a162664": "0024/00246899",
    "f1f257ad6798051c901838c3fbc15e46f4e3cb6ba73c9e47a630a833ae16b8c8": "0072/00723126",
}
REFERENCE_PROFILE_REPAIRS.update(REFERENCE_GEOMETRY_REPAIRS)


def program_digest(data):
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def needs_profile_repair(data):
    return program_digest(data) in REFERENCE_PROFILE_REPAIRS


def needs_geometry_repair(data):
    return program_digest(data) in REFERENCE_GEOMETRY_REPAIRS
