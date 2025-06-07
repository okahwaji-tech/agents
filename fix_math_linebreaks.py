#!/usr/bin/env python3
"""
Script to fix over-aggressive line breaks in mathematical expressions in markdown files.

This script identifies and fixes patterns where simple variables, short expressions,
or inline math are unnecessarily placed on separate lines, while preserving
complex equations that should remain on standalone lines.
"""

import re
import os
import argparse
from pathlib import Path
from typing import List, Tuple


class MathLineBreakFixer:
    def __init__(self):
        # Patterns for simple variables/expressions that shouldn't be on separate lines
        self.patterns = [
            # Single variables or simple expressions in display math
            (r'\n\s*\$\$([A-Za-z_]\w*)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$([A-Za-z_]\w*\s*[+\-*/]\s*[A-Za-z_]\w*)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$([A-Za-z_]\w*\s*[<>=]\s*[A-Za-z_]\w*)\$\$\s*\n', r' $\1$ '),

            # Simple matrix dimensions
            (r'\n\s*\$\$(\d+\s*\\times\s*\d+)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(n\s*\\times\s*n)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(m\s*\\times\s*n)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(p\s*\\times\s*p)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$([a-z]\s*\\times\s*[a-z])\$\$\s*\n', r' $\1$ '),

            # Simple Greek letters
            (r'\n\s*\$\$(\\lambda)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\mu)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\sigma)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\theta)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\alpha)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\beta)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\gamma)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\rho)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\epsilon)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\delta)\$\$\s*\n', r' $\1$ '),

            # Simple vectors and matrices
            (r'\n\s*\$\$(\\mathbf\{[a-zA-Z]\})\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\boldsymbol\{[a-zA-Z]\})\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$([A-Z])\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$([I])\$\$\s*\n', r' $\1$ '),

            # Simple function calls and operators
            (r'\n\s*\$\$(\\text\{[^}]+\})\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\min\([^)]+\))\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\max\([^)]+\))\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\log)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\exp)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\det)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\tr)\$\$\s*\n', r' $\1$ '),

            # Simple subscripts/superscripts
            (r'\n\s*\$\$([A-Za-z_]\w*_\{?[A-Za-z0-9]+\}?)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$([A-Za-z_]\w*\^\{?[A-Za-z0-9]+\}?)\$\$\s*\n', r' $\1$ '),

            # Simple mathematical operations
            (r'\n\s*\$\$([A-Za-z_]\w*\s*[+\-]\s*[A-Za-z_]\w*)\$\$\s*\n', r' $\1$ '),

            # Simple parenthetical expressions
            (r'\n\s*\$\$(\([A-Za-z_]\w*\s*[+\-*/]\s*[A-Za-z_]\w*\))\$\$\s*\n', r' $\1$ '),

            # Patterns in text blocks (between paragraphs)
            (r'(\w+)\s*\n\s*\$\$([A-Za-z_]\w*)\$\$\s*\n\s*(\w+)', r'\1 $\2$ \3'),

            # Fix cases where variables are split across lines in sentences
            (r'(\w+)\s+\n\s*\$\$([A-Za-z_]\w*)\$\$\s*\n\s*(\w+)', r'\1 $\2$ \3'),

            # Common mathematical constants and simple expressions
            (r'\n\s*\$\$(0)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(1)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\infty)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(\\pi)\$\$\s*\n', r' $\1$ '),
            (r'\n\s*\$\$(e)\$\$\s*\n', r' $\1$ '),

            # Additional patterns for common mathematical expressions
            (r'\n\s*\$\$([a-zA-Z]_[a-zA-Z0-9]+)\$\$\s*\n', r' $\1$ '),  # Simple subscripts
            (r'\n\s*\$\$([a-zA-Z]\^[a-zA-Z0-9]+)\$\$\s*\n', r' $\1$ '),  # Simple superscripts
            (r'\n\s*\$\$(\\mathbb\{[A-Z]\})\$\$\s*\n', r' $\1$ '),  # Blackboard bold
            (r'\n\s*\$\$(\\mathcal\{[A-Z]\})\$\$\s*\n', r' $\1$ '),  # Calligraphic
            (r'\n\s*\$\$(\\hat\{[a-zA-Z]\})\$\$\s*\n', r' $\1$ '),  # Hat notation
            (r'\n\s*\$\$(\\tilde\{[a-zA-Z]\})\$\$\s*\n', r' $\1$ '),  # Tilde notation
            (r'\n\s*\$\$(\\bar\{[a-zA-Z]\})\$\$\s*\n', r' $\1$ '),  # Bar notation
        ]
        
        # Patterns to avoid (complex expressions that should stay on separate lines)
        self.avoid_patterns = [
            r'\\sum',
            r'\\prod',
            r'\\int',
            r'\\frac\{[^}]+\}\{[^}]+\}',
            r'\\begin\{',
            r'\\end\{',
            r'\\left',
            r'\\right',
            r'=.*\\',  # Equations with backslashes (likely complex)
            r'\{.*\}.*\{.*\}',  # Multiple braces (likely complex)
            r'\\[a-zA-Z]+\{.*\\[a-zA-Z]+',  # Multiple LaTeX commands
            r'\\sqrt',
            r'\\lim',
            r'\\partial',
            r'\\nabla',
            r'\\cdot.*\\cdot',  # Multiple operations
            r'[+\-*/].*[+\-*/].*[+\-*/]',  # Multiple operations
            r'\\quad',
            r'\\qquad',
            r'\\\\',  # Line breaks in equations
            r'&.*&',  # Alignment characters
            r'\\[lr]angle',
            r'\\[lr]brace',
            r'\\[lr]bracket',
            r'\\underbrace',
            r'\\overbrace',
            r'\\stackrel',
            r'\\mathcal',
            r'\\mathbb',
            r'\\mathfrak',
            r'\\operatorname',
            r'\\DeclareMathOperator',
            r'\\newcommand',
        ]
    
    def is_complex_expression(self, expr: str) -> bool:
        """Check if an expression is complex and should remain on a separate line."""
        for pattern in self.avoid_patterns:
            if re.search(pattern, expr):
                return True
        return False
    
    def fix_content(self, content: str, dry_run: bool = False) -> Tuple[str, int, List[str]]:
        """Fix over-aggressive line breaks in the content."""
        original_content = content
        fixes_applied = 0
        changes_log = []

        for pattern, replacement in self.patterns:
            # Find all matches first to check if they're complex
            matches = list(re.finditer(pattern, content))

            for match in reversed(matches):  # Process in reverse to maintain positions
                if len(match.groups()) > 0:
                    expr = match.group(1)
                    if not self.is_complex_expression(expr):
                        old_text = match.group(0)
                        new_text = re.sub(pattern, replacement, old_text)

                        if not dry_run:
                            content = content[:match.start()] + new_text + content[match.end():]

                        changes_log.append(f"  Line ~{content[:match.start()].count(chr(10)) + 1}: '{old_text.strip()}' -> '{new_text.strip()}'")
                        fixes_applied += 1
                else:
                    if not dry_run:
                        old_content = content
                        content = re.sub(pattern, replacement, content)
                        if content != old_content:
                            fixes_applied += 1
                            changes_log.append(f"  Applied pattern: {pattern}")
                    else:
                        if re.search(pattern, content):
                            fixes_applied += 1
                            changes_log.append(f"  Would apply pattern: {pattern}")

        return content, fixes_applied, changes_log
    
    def process_file(self, file_path: Path, dry_run: bool = False) -> Tuple[bool, int]:
        """Process a single markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            fixed_content, fixes_applied, changes_log = self.fix_content(original_content, dry_run)

            if fixes_applied > 0:
                if dry_run:
                    print(f"🔍 Would fix {fixes_applied} issues in {file_path}")
                    if changes_log:
                        print("   Changes that would be made:")
                        for change in changes_log[:10]:  # Show first 10 changes
                            print(change)
                        if len(changes_log) > 10:
                            print(f"   ... and {len(changes_log) - 10} more changes")
                else:
                    # Create backup
                    backup_path = file_path.with_suffix(file_path.suffix + '.backup')
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)

                    # Write fixed content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)

                    print(f"✅ Fixed {fixes_applied} issues in {file_path}")
                    print(f"   Backup created: {backup_path}")
                    if changes_log:
                        print("   Sample changes made:")
                        for change in changes_log[:5]:  # Show first 5 changes
                            print(change)
                        if len(changes_log) > 5:
                            print(f"   ... and {len(changes_log) - 5} more changes")

                return True, fixes_applied
            else:
                print(f"ℹ️  No issues found in {file_path}")
                return False, 0

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return False, 0
    
    def process_directory(self, directory: Path, pattern: str = "*.md", dry_run: bool = False) -> None:
        """Process all markdown files in a directory."""
        total_files = 0
        total_fixes = 0
        modified_files = 0

        mode_text = "DRY RUN - " if dry_run else ""
        print(f"🔍 {mode_text}Scanning {directory} for {pattern} files...")

        for file_path in directory.rglob(pattern):
            if file_path.is_file() and not file_path.name.endswith('.backup'):
                total_files += 1
                modified, fixes = self.process_file(file_path, dry_run)
                if modified:
                    modified_files += 1
                total_fixes += fixes

        print(f"\n📊 Summary:")
        print(f"   Files processed: {total_files}")
        if dry_run:
            print(f"   Files that would be modified: {modified_files}")
            print(f"   Total fixes that would be applied: {total_fixes}")
        else:
            print(f"   Files modified: {modified_files}")
            print(f"   Total fixes applied: {total_fixes}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix over-aggressive line breaks in mathematical expressions"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to file or directory to process"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.md",
        help="File pattern to match (default: *.md)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making changes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information about patterns and fixes"
    )

    args = parser.parse_args()

    path = Path(args.path)
    fixer = MathLineBreakFixer()

    if not path.exists():
        print(f"❌ Path does not exist: {path}")
        return 1

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")

    if args.verbose:
        print(f"📋 Loaded {len(fixer.patterns)} fix patterns")
        print(f"🚫 Avoiding {len(fixer.avoid_patterns)} complex expression patterns")

    if path.is_file():
        fixer.process_file(path, args.dry_run)
    else:
        fixer.process_directory(path, args.pattern, args.dry_run)

    return 0


if __name__ == "__main__":
    exit(main())
