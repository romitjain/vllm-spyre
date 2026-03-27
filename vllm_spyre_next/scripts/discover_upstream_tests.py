#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Discover upstream vLLM tests relevant to OOT (out-of-tree) implementations.

Scans custom_ops/ for OOT class registrations and finds upstream tests that
import or instantiate those upstream classes.

Usage:
    python scripts/discover_upstream_tests.py --vllm-tests ../vllm/tests
    python scripts/discover_upstream_tests.py --vllm-tests ../vllm/tests --class SpyreRMSNorm
"""

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OOTRegistration:
    """Metadata for an OOT class registration."""

    oot_class: str  # e.g., "SpyreRMSNorm"
    upstream_class: str  # e.g., "RMSNorm"
    upstream_module: str  # e.g., "vllm.model_executor.layers.layernorm"
    file_path: Path


@dataclass
class TestMatch:
    """A test file that matches an OOT registration."""

    test_file: Path
    imports_class: bool
    instantiates_class: bool
    test_functions: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        if self.imports_class and self.instantiates_class:
            return "direct"
        return "integration"

    @property
    def test_count(self) -> int:
        return len(self.test_functions)


def has_register_oot_decorator(node: ast.ClassDef) -> bool:
    """Check if a class has a @*.register_oot decorator."""
    for decorator in node.decorator_list:
        # Handle @X.register_oot, @X.register_oot(), @X.register_oot(name="Y")
        if isinstance(decorator, ast.Call):
            func = decorator.func
        else:
            func = decorator

        if isinstance(func, ast.Attribute) and func.attr == "register_oot":
            return True
    return False


def get_base_class_name(node: ast.ClassDef) -> str | None:
    """Extract the first base class name from a class definition."""
    if not node.bases:
        return None

    base = node.bases[0]
    if isinstance(base, ast.Name):
        return base.id
    elif isinstance(base, ast.Attribute):
        return base.attr
    return None


def find_import_module(tree: ast.Module, class_name: str) -> str | None:
    """Find the module path for an imported class name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == class_name:
                    return node.module
    return None


def find_oot_classes(custom_ops_dir: Path) -> list[OOTRegistration]:
    """Parse custom_ops/ to find all OOT class registrations."""
    registrations = []

    for py_file in custom_ops_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue

        try:
            source = py_file.read_text()
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            if not has_register_oot_decorator(node):
                continue

            base_class = get_base_class_name(node)
            if not base_class:
                continue

            upstream_module = find_import_module(tree, base_class)
            if not upstream_module:
                continue

            registrations.append(
                OOTRegistration(
                    oot_class=node.name,
                    upstream_class=base_class,
                    upstream_module=upstream_module,
                    file_path=py_file,
                )
            )

    return registrations


def check_imports_class(tree: ast.Module, class_name: str) -> bool:
    """Check if the AST imports a given class name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == class_name:
                    return True
    return False


def check_instantiates_class(tree: ast.Module, class_name: str) -> bool:
    """Check if the AST contains a call that instantiates the class."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Direct call: ClassName(...)
            if isinstance(func, ast.Name) and func.id == class_name:
                return True
            # Attribute call: module.ClassName(...)
            if isinstance(func, ast.Attribute) and func.attr == class_name:
                return True
    return False


def extract_test_functions(tree: ast.Module) -> list[str]:
    """Extract all test function names from the AST."""
    test_functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            test_functions.append(node.name)
    return test_functions


def find_tests_using_class(class_name: str, tests_dir: Path) -> list[TestMatch]:
    """Find all test files that import or instantiate the given class."""
    matches = []

    for test_file in tests_dir.rglob("test_*.py"):
        try:
            source = test_file.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        imports = check_imports_class(tree, class_name)
        instantiates = check_instantiates_class(tree, class_name)

        if imports or instantiates:
            test_functions = extract_test_functions(tree)
            matches.append(
                TestMatch(
                    test_file=test_file,
                    imports_class=imports,
                    instantiates_class=instantiates,
                    test_functions=test_functions,
                )
            )

    return matches


def print_results(
    registrations: list[OOTRegistration],
    results: dict[str, list[TestMatch]],
    tests_dir: Path,
) -> None:
    """Print discovered tests grouped by OOT class and confidence."""
    print("Discovering upstream tests for OOT implementations...\n")

    total_files = 0
    total_tests = 0

    for reg in registrations:
        matches = results.get(reg.oot_class, [])
        print(f"{reg.oot_class} (replaces {reg.upstream_class} from {reg.upstream_module})")

        direct = [m for m in matches if m.confidence == "direct"]
        integration = [m for m in matches if m.confidence == "integration"]

        reg_files = len(matches)
        reg_tests = sum(m.test_count for m in matches)
        total_files += reg_files
        total_tests += reg_tests

        if direct:
            direct_tests = sum(m.test_count for m in direct)
            print(f"  Direct tests ({len(direct)} files, {direct_tests} test functions):")
            for match in direct:
                rel_path = match.test_file.relative_to(tests_dir)
                print(f"    {rel_path}")
                for func in match.test_functions:
                    print(f"      - {func}")

        if integration:
            integration_tests = sum(m.test_count for m in integration)
            print(f"  Integration tests ({len(integration)} files, {integration_tests} test functions):")
            for match in integration:
                rel_path = match.test_file.relative_to(tests_dir)
                print(f"    {rel_path}")
                for func in match.test_functions:
                    print(f"      - {func}")

        if not matches:
            print("  No tests found")

        print()

    print(f"Summary: {total_files} test files, {total_tests} test functions")


def main():
    parser = argparse.ArgumentParser(
        description="Discover upstream vLLM tests for OOT implementations"
    )
    parser.add_argument(
        "--vllm-tests",
        dest="vllm_tests",
        required=True,
        help="Path to upstream vLLM tests directory (required)",
    )
    parser.add_argument(
        "--class",
        dest="oot_class",
        help="Filter to a specific OOT class (e.g., SpyreRMSNorm)",
    )
    args = parser.parse_args()

    # Validate vllm tests directory
    tests_dir = Path(args.vllm_tests)
    if not tests_dir.exists():
        print(f"Error: vLLM tests directory does not exist: {tests_dir}", file=sys.stderr)
        sys.exit(1)
    if not tests_dir.is_dir():
        print(f"Error: vLLM tests path is not a directory: {tests_dir}", file=sys.stderr)
        sys.exit(1)

    # Find custom_ops directory
    script_dir = Path(__file__).parent
    custom_ops_dir = script_dir.parent / "vllm_spyre_next" / "custom_ops"

    if not custom_ops_dir.exists():
        print(f"Error: custom_ops directory not found at {custom_ops_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Custom ops dir: {custom_ops_dir}")
    print(f"Upstream tests dir: {tests_dir}\n")

    # Parse OOT registrations
    registrations = find_oot_classes(custom_ops_dir)

    if not registrations:
        print("Error: No OOT registrations found in custom_ops/", file=sys.stderr)
        sys.exit(1)

    if args.oot_class:
        filtered = [r for r in registrations if r.oot_class == args.oot_class]
        if not filtered:
            available = ", ".join(r.oot_class for r in registrations)
            print(
                f"Error: OOT class '{args.oot_class}' not found. Available: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        registrations = filtered

    # Find matching tests for each registration
    results: dict[str, list[TestMatch]] = {}
    for reg in registrations:
        matches = find_tests_using_class(reg.upstream_class, tests_dir)
        results[reg.oot_class] = matches

    # Print results
    print_results(registrations, results, tests_dir)


if __name__ == "__main__":
    main()
