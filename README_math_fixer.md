# Math Line Break Fixer

A Python script to systematically fix over-aggressive line breaks in mathematical expressions in markdown files.

## Features

- **Smart Pattern Detection**: Identifies 40+ patterns of unnecessary line breaks
- **Complexity Awareness**: Preserves complex equations that should remain on separate lines
- **Safety First**: Dry-run mode, automatic backups, detailed logging
- **Flexible**: Works on single files or entire directories

## Usage

### Basic Usage

```bash
# Check what would be fixed (dry-run)
python3 fix_math_linebreaks.py docs/materials/math/eigenvalues-eigenvectors.md --dry-run

# Actually fix the file
python3 fix_math_linebreaks.py docs/materials/math/eigenvalues-eigenvectors.md

# Process entire directory
python3 fix_math_linebreaks.py docs/materials/math/ --dry-run
python3 fix_math_linebreaks.py docs/materials/math/
```

### Advanced Options

```bash
# Verbose output with pattern details
python3 fix_math_linebreaks.py docs/ --dry-run --verbose

# Custom file pattern
python3 fix_math_linebreaks.py docs/ --pattern "*.markdown" --dry-run
```

## What It Fixes

### ✅ Fixes These Patterns:
- Single variables: `$$A$$` → `$A$`
- Simple dimensions: `$$n \times n$$` → `$n \times n$`
- Greek letters: `$$\lambda$$` → `$\lambda$`
- Simple subscripts: `$$A_i$$` → `$A_i$`
- Common constants: `$$\pi$$` → `$\pi$`

### 🚫 Preserves These (Complex Expressions):
- Fractions: `$$\frac{a}{b}$$`
- Integrals: `$$\int f(x) dx$$`
- Multi-line equations with `\\`
- Alignment structures with `&`
- Complex LaTeX commands

## Example Output

```
🔍 DRY RUN MODE - No files will be modified
✅ Would fix 16 issues in docs/materials/math/eigenvalues-eigenvectors.md
   Changes that would be made:
     Line ~329: '$$A$$' → '$A$'
     Line ~331: '$$n \times n$$' → '$n \times n$'
     Line ~335: '$$\lambda$$' → '$\lambda$'
     ... and 13 more changes

📊 Summary:
   Files processed: 1
   Files that would be modified: 1
   Total fixes that would be applied: 16
```

## Safety Features

1. **Automatic Backups**: Creates `.backup` files before making changes
2. **Dry-Run Mode**: Preview all changes before applying them
3. **Pattern Validation**: Only fixes simple expressions, preserves complex ones
4. **Detailed Logging**: Shows exactly what changes are made

## Installation

No additional dependencies required beyond Python 3.6+ standard library.

## Results

- ✅ Fixed 16 over-aggressive line breaks in eigenvalues-eigenvectors.md
- ✅ Verified all other documentation files are properly formatted
- ✅ No additional files need fixing in current documentation
