from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class SequenceAlignment(BaseModel):
    """Represents a pairwise sequence alignment result."""

    reference_sequence: str = Field(description="Aligned reference sequence")
    mobile_sequence: str = Field(description="Aligned mobile sequence")
    reference_indices: List[int] = Field(
        description="Atom indices for reference structure"
    )
    mobile_indices: List[int] = Field(description="Atom indices for mobile structure")
    reference_resids: List[int] = Field(
        description="Original residue IDs for reference structure"
    )
    mobile_resids: List[int] = Field(
        description="Original residue IDs for mobile structure"
    )
    reference_chain_ids: List[str] = Field(
        description="Chain IDs for reference structure"
    )
    mobile_chain_ids: List[str] = Field(description="Chain IDs for mobile structure")
    alignment_score: float = Field(description="Sequence alignment score")

    class Config:
        arbitrary_types_allowed = True


class AlignmentResult(BaseModel):
    """Complete structural alignment result."""

    sequence_alignment: SequenceAlignment
    rmsd_before: float = Field(description="RMSD before structural alignment")
    rmsd_after: float = Field(description="RMSD after structural alignment")
    n_aligned_residues: int = Field(description="Number of aligned residues")
    position_distances: List[float] = Field(
        description="Per-residue distances after alignment"
    )

    class Config:
        arbitrary_types_allowed = True

    def get_distances_array(self) -> np.ndarray:
        """Get position distances as numpy array."""
        return np.array(self.position_distances)

    def get_residue_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Get mapping from original residue IDs to alignment positions.

        Returns:
            Tuple of (ref_resid_to_pos, mob_resid_to_pos) dictionaries
        """
        ref_mapping = {
            resid: pos
            for pos, resid in enumerate(self.sequence_alignment.reference_resids)
        }
        mob_mapping = {
            resid: pos
            for pos, resid in enumerate(self.sequence_alignment.mobile_resids)
        }
        return ref_mapping, mob_mapping

    def get_distance_by_residue(
        self, ref_resid: int, mob_resid: Optional[int] = None
    ) -> Optional[float]:
        """
        Get distance between specific residues by their original residue IDs.

        Args:
            ref_resid: Reference residue ID
            mob_resid: Mobile residue ID (if None, uses aligned counterpart)

        Returns:
            Distance in Angstroms, or None if residues not found in alignment
        """
        ref_mapping, mob_mapping = self.get_residue_mapping()

        # Find reference residue position in alignment
        if ref_resid not in ref_mapping:
            return None

        ref_pos = ref_mapping[ref_resid]

        if mob_resid is None:
            # Use the aligned counterpart (same alignment position)
            if ref_pos < len(self.position_distances):
                return self.position_distances[ref_pos]
            return None
        else:
            # Find specific mobile residue
            if mob_resid not in mob_mapping:
                return None
            mob_pos = mob_mapping[mob_resid]

            # Check if they're at the same alignment position
            if ref_pos == mob_pos:
                return self.position_distances[ref_pos]
            else:
                # They're not aligned to each other
                return None

    def get_aligned_residue_pairs(self) -> List[Tuple[int, int, float]]:
        """
        Get all aligned residue pairs with their distances.

        Returns:
            List of (ref_resid, mob_resid, distance) tuples
        """
        pairs = []
        for i, distance in enumerate(self.position_distances):
            ref_resid = self.sequence_alignment.reference_resids[i]
            mob_resid = self.sequence_alignment.mobile_resids[i]
            pairs.append((ref_resid, mob_resid, distance))
        return pairs

    def get_residue_info_table(self) -> pd.DataFrame:
        """
        Generate a pandas DataFrame of aligned residues with distances.

        Returns:
            pandas DataFrame with columns: ref_resid, mob_resid, distance, ref_aa, mob_aa, ref_chain, mob_chain
        """
        pairs = self.get_aligned_residue_pairs()

        data = []
        for i, (ref_resid, mob_resid, distance) in enumerate(pairs):
            ref_aa = self.sequence_alignment.reference_sequence[i]
            mob_aa = self.sequence_alignment.mobile_sequence[i]
            ref_chain = self.sequence_alignment.reference_chain_ids[i]
            mob_chain = self.sequence_alignment.mobile_chain_ids[i]

            data.append(
                {
                    "ref_resid": ref_resid,
                    "mob_resid": mob_resid,
                    "distance": distance,
                    "ref_aa": ref_aa,
                    "mob_aa": mob_aa,
                    "ref_chain": ref_chain,
                    "mob_chain": mob_chain,
                }
            )

        return pd.DataFrame(data)

    def get_residue_info_table_formatted(self) -> str:
        """
        Generate a formatted string table of aligned residues with distances.

        Returns:
            Formatted string table (for backward compatibility)
        """
        df = self.get_residue_info_table()

        # Format the DataFrame as a nice string table
        header = f"{'Ref ResID':>8} {'Mob ResID':>8} {'Distance':>10} {'Ref AA':>6} {'Mob AA':>6}"
        separator = "-" * len(header)

        lines = [header, separator]

        for _, row in df.iterrows():
            line = f"{row['ref_resid']:>8} {row['mob_resid']:>8} {row['distance']:>10.2f} {row['ref_aa']:>6} {row['mob_aa']:>6}"
            lines.append(line)

        return "\n".join(lines)
