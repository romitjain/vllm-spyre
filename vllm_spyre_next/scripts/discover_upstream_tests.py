#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Discover upstream vLLM tests relevant to OOT (out-of-tree) implementations.

Scans custom_ops/ for OOT class registrations and finds upstream tests that
import or instantiate those upstream classes.

Usage:
    python scripts/discover_upstream_tests.py
    python scripts/discover_upstream_tests.py --class SpyreRMSNorm
    python scripts/discover_upstream_tests.py --upstream-tests ../vllm/tests
"""

import argparse
import ast
from dataclasses import dataclass
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

    @property
    def confidence(self) -> str:
        if self.imports_class and self.instantiates_class:
            return "direct"
        return "integration"


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
            matches.append(
                TestMatch(
                    test_file=test_file,
                    imports_class=imports,
                    instantiates_class=instantiates,
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

    for reg in registrations:
        matches = results.get(reg.oot_class, [])
        print(f"{reg.oot_class} (replaces {reg.upstream_class} from {reg.upstream_module})")

        direct = [m for m in matches if m.confidence == "direct"]
        integration = [m for m in matches if m.confidence == "integration"]

        if direct:
            print("  Direct tests (high confidence):")
            for match in direct:
                rel_path = match.test_file.relative_to(tests_dir)
                print(f"    {rel_path}")

        if integration:
            print("  Integration tests (medium confidence):")
            for match in integration:
                rel_path = match.test_file.relative_to(tests_dir)
                print(f"    {rel_path}")

        if not matches:
            print("  No tests found")

        print()


def resolve_upstream_tests_dir(args_path: str | None) -> Path:
    """Resolve the upstream tests directory."""
    if args_path:
        return Path(args_path)

    # Try ../vllm/tests relative to this script
    script_dir = Path(__file__).parent
    vllm_tests = script_dir.parent.parent.parent / "vllm" / "tests"
    if vllm_tests.exists():
        return vllm_tests

    # Try cached location
    cache_dir = Path.home() / ".cache" / "vllm-upstream-tests"
    if cache_dir.exists():
        # Find the first subdirectory with tests
        for subdir in cache_dir.iterdir():
            tests_path = subdir / "tests"
            if tests_path.exists():
                return tests_path

    raise FileNotFoundError(
        "Could not find upstream tests directory. "
        "Use --upstream-tests to specify the path."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Discover upstream vLLM tests for OOT implementations"
    )
    parser.add_argument(
        "--class",
        dest="oot_class",
        help="Filter to a specific OOT class (e.g., SpyreRMSNorm)",
    )
    parser.add_argument(
        "--upstream-tests",
        dest="upstream_tests",
        help="Path to upstream vLLM tests directory",
    )
    args = parser.parse_args()

    # Find custom_ops directory
    script_dir = Path(__file__).parent
    custom_ops_dir = script_dir.parent / "vllm_spyre_next" / "custom_ops"

    if not custom_ops_dir.exists():
        print(f"Error: custom_ops directory not found at {custom_ops_dir}")
        return 1

    # Find upstream tests directory
    try:
        tests_dir = resolve_upstream_tests_dir(args.upstream_tests)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"Custom ops dir: {custom_ops_dir}")
    print(f"Upstream tests dir: {tests_dir}\n")

    # Parse OOT registrations
    registrations = find_oot_classes(custom_ops_dir)

    if args.oot_class:
        registrations = [r for r in registrations if r.oot_class == args.oot_class]

    if not registrations:
        print("No OOT registrations found")
        return 0

    # Find matching tests for each registration
    results: dict[str, list[TestMatch]] = {}
    for reg in registrations:
        matches = find_tests_using_class(reg.upstream_class, tests_dir)
        results[reg.oot_class] = matches

    # Print results
    print_results(registrations, results, tests_dir)

    return 0


if __name__ == "__main__":
    exit(main())
