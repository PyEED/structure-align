# Structure Align

A minimal Python package for structural alignment of protein structures with different lengths.

## Overview

Structure Align performs pairwise sequence alignment followed by structural alignment of protein structures that don't have the same number of amino acids. It's designed to be simple, clean, and efficient for structural biology applications.

## Key Features

- **Sequence-based alignment**: Uses BioPython's pairwise aligner to find matching residues
- **Flexible selection**: Support for different atom selections (CA, backbone, etc.)
- **Structured results**: Uses Pydantic models for clean, validated data structures
- **RMSD calculation**: Provides before/after RMSD values
- **Position-wise analysis**: Calculate per-residue distances after alignment
- **Residue ID mapping**: Query distances by original residue IDs (e.g., "residue 22")
- **Gap handling**: Automatically handles sequence alignment gaps
- **Visualization**: Built-in plotting functionality for distance analysis
- **Interactive plots**: Rich hover information with Plotly integration
- **Chain-aware display**: Visual separation and coloring of different protein chains

## Installation

```bash
# Install dependencies
poetry install

# Or with pip (after building)
pip install structure-align
```

## Quick Start

```python
import MDAnalysis as mda
from structure_align import StructuralAligner

# Load your structures
reference = mda.Universe("reference.pdb")
mobile = mda.Universe("mobile.pdb")

# Initialize aligner
aligner = StructuralAligner()

# Perform alignment
result = aligner.align(reference, mobile, selection="name CA")

# Print results
print(f"RMSD: {result.rmsd_before:.2f} → {result.rmsd_after:.2f} Å")
print(f"Aligned residues: {result.n_aligned_residues}")

# Query specific residue distances
distance = result.get_distance_by_residue(22)  # Distance for residue 22
if distance:
    print(f"Residue 22 distance: {distance:.2f} Å")
```

## Residue ID Mapping

One of the key features is the ability to query distances by original residue IDs, even after sequence alignment with gaps:

```python
# Get distance for specific residue
distance = result.get_distance_by_residue(22)

# Get all aligned residue pairs
pairs = result.get_aligned_residue_pairs()
for ref_resid, mob_resid, distance in pairs:
    print(f"Ref {ref_resid} ↔ Mob {mob_resid}: {distance:.2f} Å")

# Get formatted table of results
df = result.get_residue_info_table()  # Returns pandas DataFrame
print(df.head())

# Or get formatted string (backward compatibility)
table_str = result.get_residue_info_table_formatted()
print(table_str)

# Get residue mappings
ref_mapping, mob_mapping = result.get_residue_mapping()
```

## API Reference

### StructuralAligner

Main class for performing structural alignments.

#### Methods

- `__init__(gap_open=-10.0, gap_extend=-0.5)`: Initialize with gap penalties
- `align(reference, mobile, selection="name CA")`: Perform alignment
- `calculate_position_distances(result)`: Get per-residue distances
- `plot_distances(result, **kwargs)`: Create distance plot

### AlignmentResult

Complete alignment result with residue mapping capabilities.

#### Key Methods

- `get_distance_by_residue(ref_resid)`: Get distance for specific residue ID
- `get_aligned_residue_pairs()`: Get all (ref_resid, mob_resid, distance) tuples
- `get_residue_mapping()`: Get residue ID to position mappings
- `get_residue_info_table()`: Get formatted table of aligned residues

#### Properties

- `rmsd_before/rmsd_after`: RMSD values before and after alignment
- `n_aligned_residues`: Number of successfully aligned residues
- `position_distances`: Per-position distances after alignment

## Examples

### Basic Usage

```python
from structure_align import StructuralAligner
import MDAnalysis as mda

# Load structures
ref = mda.Universe("protein1.pdb")
mob = mda.Universe("protein2.pdb")

# Align using CA atoms
aligner = StructuralAligner()
result = aligner.align(ref, mob)

print(f"RMSD: {result.rmsd_before:.2f} → {result.rmsd_after:.2f} Å")
```

### Query Specific Residues

```python
# Query distance for residue 22
distance = result.get_distance_by_residue(22)
if distance:
    print(f"Residue 22: {distance:.2f} Å")
else:
    print("Residue 22 not found in alignment")

# Find high-distance residues
pairs = result.get_aligned_residue_pairs()
high_distance = [(r1, r2, d) for r1, r2, d in pairs if d > 3.0]
print(f"Found {len(high_distance)} residues with distance > 3.0 Å")
```

### Analysis and Visualization

```python
# Get comprehensive analysis as DataFrame
df = result.get_residue_info_table()
print(f"Alignment shape: {df.shape}")
print(df.describe())

# Find high-distance residues
high_distance = df[df['distance'] > 3.0]
print(f"Found {len(high_distance)} residues with distance > 3.0 Å")

# Find specific amino acid combinations
cys_pairs = df[(df['ref_aa'] == 'C') & (df['mob_aa'] == 'C')]
print("Cysteine-Cysteine alignments:")
print(cys_pairs[['ref_resid', 'mob_resid', 'distance']])

# Sort by distance
worst_aligned = df.nlargest(10, 'distance')
print("10 worst aligned residues:")
print(worst_aligned)

# Plot distances with residue information
fig = aligner.plot_distances(result, title="Residue Distance Analysis")
fig.show()

# Statistical analysis
distances = result.get_distances_array()
print(f"Mean distance: {distances.mean():.2f} Å")
print(f"Std deviation: {distances.std():.2f} Å")
```

### DataFrame Operations

The `get_residue_info_table()` method returns a pandas DataFrame with the following columns:
- `ref_resid`: Reference residue ID
- `mob_resid`: Mobile residue ID  
- `distance`: Distance between aligned residues (Å)
- `ref_aa`: Reference amino acid (single letter)
- `mob_aa`: Mobile amino acid (single letter)

This enables powerful analysis:

```python
df = result.get_residue_info_table()

# Filter by distance threshold
high_rmsd = df[df['distance'] > 2.0]

# Filter by amino acid type
aromatics = df[df['ref_aa'].isin(['F', 'W', 'Y'])]

# Group by amino acid and get statistics
aa_stats = df.groupby('ref_aa')['distance'].agg(['mean', 'std', 'count'])

# Export to CSV for further analysis
df.to_csv('alignment_results.csv', index=False)

# Merge with other data
# df = df.merge(other_data, on='ref_resid')
```

## Handling Sequence Gaps

The package automatically handles sequence alignment gaps:

1. **Sequence alignment**: Creates optimal alignment with gaps (-)
2. **Residue mapping**: Only aligned residues (no gaps) are used for structural alignment
3. **ID preservation**: Original residue IDs are preserved for querying
4. **Gap tracking**: You can see which residues were aligned vs. skipped

```python
# Example with gaps
# Reference: ACDEFGHIK
# Mobile:    A-DEF-HIK
# Result:    Only ADEF and HIK positions are structurally aligned
# But you can still query by original residue IDs
```

## Dependencies

- MDAnalysis: Structure handling and analysis
- BioPython: Sequence alignment
- NumPy: Numerical calculations
- Pandas: Data analysis and DataFrame operations
- Matplotlib: Static plotting
- Plotly: Interactive plotting with rich hover information
- Pydantic: Data validation and models

## License

MIT License

### Interactive Plotting

The package provides both static (matplotlib) and interactive (Plotly) plotting options:

```python
# Static matplotlib plot
fig_static = aligner.plot_distances(result)
fig_static.show()

# Interactive Plotly plot with rich hover information
fig_interactive = aligner.plot_distances_interactive(result)
fig_interactive.show()

# Save interactive plot as HTML
fig_interactive.write_html("interactive_plot.html")
```

#### Interactive Features

**Rich Hover Tooltips**: Each point shows:
- Residue information: `LEU123 (Chain A)`
- Amino acid conservation
- Precise distance measurements
- Alignment position

**Interactive Controls**:
- 🔍 **Zoom**: Click and drag to zoom into regions
- 🖱️ **Pan**: Shift+drag to navigate
- 📏 **Range Slider**: Navigate large proteins easily
- 🎨 **Legend**: Show/hide specific chains
- 💾 **Export**: HTML format for sharing

**Chain Visualization**:
- Color-coded chains with boundaries
- Custom color schemes
- Chain-specific statistics

```python
# Custom chain colors
colors = {'A': 'red', 'B': 'blue', 'C': 'green'}
fig = aligner.plot_distances_interactive(result, chain_colors=colors)

# Large plot for detailed analysis
fig = aligner.plot_distances_interactive(result, height=800, width=1400)
```

### Analysis and Visualization
