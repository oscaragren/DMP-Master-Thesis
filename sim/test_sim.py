#!/usr/bin/env python3
"""
Test script to validate left_arm.urdf
Verifies:
  1. Valid XML parsing
  2. jLeftShoulder_rotz is 'fixed'
  3. jLeftArm_rotz exists, is 'revolute', axis Z, parent=left_upper_arm, child=left_arm_rotz
  4. jLeftElbow_roty parent is left_arm_rotz (not left_upper_arm)
  5. No orphan links / continuous chain to fingers
"""

import xml.etree.ElementTree as ET
import sys
import os

# Relative path to arm/ directory
URDF_FILE = os.path.join(os.path.dirname(__file__), "..", "arm", "left_arm.urdf")

def parse(file):
    try:
        tree = ET.parse(file)
        print(f"[OK] XML parsing: {file}")
        return tree.getroot()
    except ET.ParseError as e:
        print(f"[FAIL] XML parsing: {e}")
        sys.exit(1)

def get_joints(root):
    return {j.get("name"): j for j in root.findall("joint")}

def get_links(root):
    return {l.get("name") for l in root.findall("link")}

def check(condition, msg_ok, msg_fail):
    if condition:
        print(f"[OK] {msg_ok}")
    else:
        print(f"[FAIL] {msg_fail}")
    return condition

def main():
    root = parse(URDF_FILE)
    joints = get_joints(root)
    links = get_links(root)
    all_ok = True

    print("\n--- TEST 1: jLeftShoulder_rotz is fixed ---")
    j = joints.get("jLeftShoulder_rotz")
    all_ok &= check(j is not None, "joint found", "jLeftShoulder_rotz not found")
    if j is not None:
        all_ok &= check(j.get("type") == "fixed",
                        "type=fixed",
                        f"type={j.get('type')} (expected: fixed)")

    print("\n--- TEST 2: jLeftArm_rotz exists and is correctly defined ---")
    j = joints.get("jLeftArm_rotz")
    all_ok &= check(j is not None, "joint jLeftArm_rotz found", "jLeftArm_rotz NOT FOUND")
    if j is not None:
        all_ok &= check(j.get("type") == "revolute",
                        "type=revolute",
                        f"type={j.get('type')} (expected: revolute)")

        parent = j.find("parent")
        all_ok &= check(parent is not None and parent.get("link") == "left_upper_arm",
                        "parent=left_upper_arm",
                        f"parent={parent.get('link') if parent is not None else 'None'}")

        child = j.find("child")
        all_ok &= check(child is not None and child.get("link") == "left_arm_rotz",
                        "child=left_arm_rotz",
                        f"child={child.get('link') if child is not None else 'None'}")

        axis = j.find("axis")
        all_ok &= check(axis is not None and axis.get("xyz") in ("0 0 -1", "0 0 1"),
                        f"axis Z = {axis.get('xyz') if axis is not None else 'None'}",
                        f"unexpected axis: {axis.get('xyz') if axis is not None else 'None'}")

        origin = j.find("origin")
        all_ok &= check(origin is not None and "-0.305" in origin.get("xyz", ""),
                        f"origin xyz={origin.get('xyz') if origin is not None else 'None'} (contains -0.305)",
                        f"unexpected origin xyz: {origin.get('xyz') if origin is not None else 'None'}")

    print("\n--- TEST 3: jLeftElbow_roty is connected to left_arm_rotz ---")
    j = joints.get("jLeftElbow_roty")
    all_ok &= check(j is not None, "joint found", "jLeftElbow_roty not found")
    if j is not None:
        parent = j.find("parent")
        all_ok &= check(parent is not None and parent.get("link") == "left_arm_rotz",
                        "parent=left_arm_rotz",
                        f"parent={parent.get('link') if parent is not None else 'None'} (expected: left_arm_rotz)")

        origin = j.find("origin")
        all_ok &= check(origin is not None and origin.get("xyz") == "0 0 0",
                        "origin xyz=0 0 0 (offset absorbed by jLeftArm_rotz)",
                        f"origin xyz={origin.get('xyz') if origin is not None else 'None'}")

    print("\n--- TEST 4: Complete chain to fingers ---")
    expected_chain = [
        ("jLeftShoulder_rotz",      "arm_base_rotated",     "left_shoulder_rotz"),
        ("jLeftShoulder_rotx",      "left_shoulder_rotz",   "left_shoulder_rotx"),
        ("jLeftShoulder_roty",      "left_shoulder_rotx",   "left_shoulder_rotz_arm"),
        ("jLeftShoulder_rotz_arm",  "left_shoulder_rotz_arm", "left_upper_arm"),
        ("jLeftArm_rotz",           "left_upper_arm",       "left_arm_rotz"),
        ("jLeftElbow_roty",         "left_arm_rotz",        "left_elbow_roty"),
        ("jLeftElbow_rotz",         "left_elbow_roty",      "left_forearm"),
        ("jLeftWrist_rotx",         "left_forearm",         "left_wrist_rotx"),
        ("jLeftWrist_rotz",         "left_wrist_rotx",      "left_hand"),
    ]
    for jname, expected_parent, expected_child in expected_chain:
        j = joints.get(jname)
        if j is None:
            print(f"[FAIL] {jname} not found")
            all_ok = False
            continue
        p = j.find("parent")
        c = j.find("child")
        p_ok = p is not None and p.get("link") == expected_parent
        c_ok = c is not None and c.get("link") == expected_child
        all_ok &= check(p_ok and c_ok,
                        f"{jname}: {expected_parent} → {expected_child}",
                        f"{jname}: expected {expected_parent}→{expected_child}, "
                        f"found {p.get('link') if p is not None else '?'}→{c.get('link') if c is not None else '?'}")

    print("\n--- TEST 5: link left_arm_rotz exists ---")
    all_ok &= check("left_arm_rotz" in links,
                    "left_arm_rotz present in links",
                    "left_arm_rotz MISSING from links")

    print("\n" + "="*50)
    if all_ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — see details above")
    print("="*50)
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()