"""
Tests for the StructuralAligner class.
"""

import numpy as np
import pytest

from structure_align import AlignmentResult, SequenceAlignment, StructuralAligner


class TestStructuralAligner:
    """Test cases for StructuralAligner."""

    def test_initialization(self):
        """Test aligner initialization."""
        aligner = StructuralAligner()
        assert aligner.aligner is not None
        assert aligner.aligner.open_gap_score == -10.0
        assert aligner.aligner.extend_gap_score == -0.5

        # Test custom parameters
        aligner_custom = StructuralAligner(gap_open=-5.0, gap_extend=-1.0)
        assert aligner_custom.aligner.open_gap_score == -5.0
        assert aligner_custom.aligner.extend_gap_score == -1.0

    def test_sequence_alignment(self):
        """Test sequence alignment functionality."""
        aligner = StructuralAligner()

        # Test with simple sequences
        ref_seq = "ACDEF"
        mob_seq = "ADEF"  # Missing C

        aligned_ref, aligned_mob, score = aligner._perform_sequence_alignment(
            ref_seq, mob_seq
        )

        assert len(aligned_ref) == len(aligned_mob)
        assert isinstance(score, float)

    def test_matching_indices(self):
        """Test matching indices extraction."""
        aligner = StructuralAligner()

        aligned_ref = "AC-DEF"
        aligned_mob = "A-TDEF"
        ref_indices = [0, 1, 2, 3, 4]
        mob_indices = [10, 11, 12, 13]

        matched_ref, matched_mob = aligner._get_matching_indices(
            aligned_ref, aligned_mob, ref_indices, mob_indices
        )

        # Should match A, D, E, F positions
        assert len(matched_ref) == len(matched_mob)
        assert len(matched_ref) == 4  # A, D, E, F


def test_alignment_result_model():
    """Test AlignmentResult Pydantic model."""
    seq_alignment = SequenceAlignment(
        reference_sequence="ACDEF",
        mobile_sequence="ADEF",
        reference_indices=[0, 1, 2, 3, 4],
        mobile_indices=[10, 11, 12, 13],
        reference_resids=[100, 101, 102, 103, 104],
        mobile_resids=[200, 201, 202, 203],
        alignment_score=25.0,
    )

    result = AlignmentResult(
        sequence_alignment=seq_alignment,
        rmsd_before=10.5,
        rmsd_after=2.1,
        n_aligned_residues=4,
        position_distances=[1.2, 2.3, 1.8, 2.9],
    )

    assert result.rmsd_before == 10.5
    assert result.rmsd_after == 2.1
    assert result.n_aligned_residues == 4
    assert len(result.position_distances) == 4

    distances_array = result.get_distances_array()
    assert isinstance(distances_array, np.ndarray)
    assert len(distances_array) == 4

    # Test residue mapping functionality
    ref_mapping, mob_mapping = result.get_residue_mapping()
    assert ref_mapping[100] == 0  # First residue
    assert mob_mapping[200] == 0  # First mobile residue

    # Test distance query by residue ID
    distance = result.get_distance_by_residue(100)  # First aligned residue
    assert distance == 1.2

    # Test aligned residue pairs
    pairs = result.get_aligned_residue_pairs()
    assert len(pairs) == 4
    assert pairs[0] == (100, 200, 1.2)

    # Test residue info table
    df = result.get_residue_info_table()
    assert isinstance(
        df, type(result.get_residue_info_table())
    )  # Check it's a DataFrame
    assert len(df) == 4  # Should have 4 aligned residues
    assert list(df.columns) == [
        "ref_resid",
        "mob_resid",
        "distance",
        "ref_aa",
        "mob_aa",
    ]
    assert df.iloc[0]["ref_resid"] == 100
    assert df.iloc[0]["mob_resid"] == 200
    assert df.iloc[0]["distance"] == 1.2

    # Test formatted table (backward compatibility)
    table = result.get_residue_info_table_formatted()
    assert isinstance(table, str)
    assert "Ref ResID" in table
    assert "100" in table


if __name__ == "__main__":
    pytest.main([__file__])
