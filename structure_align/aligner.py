from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from Bio import Align
from MDAnalysis.analysis.align import alignto
from MDAnalysis.lib.util import convert_aa_code

from .models import AlignmentResult, SequenceAlignment


class StructuralAligner:
    """
    Main class for structural alignment of protein structures with different lengths.

    This class handles pairwise sequence alignment followed by structural alignment
    of the matched residues.
    """

    def __init__(
        self, gap_open: float = -10.0, gap_extend: float = -0.5, verbose: bool = False
    ):
        """
        Initialize the structural aligner.

        Args:
            gap_open: Gap opening penalty for sequence alignment
            gap_extend: Gap extension penalty for sequence alignment
            verbose: Whether to print debug information during alignment
        """
        self.aligner = Align.PairwiseAligner()
        self.aligner.open_gap_score = gap_open
        self.aligner.extend_gap_score = gap_extend
        self.verbose = verbose

    def _extract_sequence_and_indices(
        self, universe: mda.Universe, selection: str
    ) -> Tuple[str, List[int], List[int], List[str]]:
        """
        Extract amino acid sequence, corresponding atom indices, residue IDs, and chain IDs from a universe.

        Args:
            universe: MDAnalysis Universe object
            selection: Selection string for atoms (e.g., 'name CA', 'backbone')

        Returns:
            Tuple of (sequence string, list of atom indices, list of residue IDs, list of chain IDs)
        """
        sequence = []
        indices = []
        resids = []
        chain_ids = []

        for segment in universe.segments:
            for residue in sorted(segment.residues, key=lambda x: x.resid):
                try:
                    # Convert residue name to single letter amino acid code
                    aa = convert_aa_code(residue.resname)
                except ValueError:
                    # Skip non-standard residues
                    continue

                # Select atoms based on selection string
                selected_atoms = residue.atoms.select_atoms(selection)
                if len(selected_atoms) > 0:
                    sequence.append(aa)
                    # Use the first selected atom's index (e.g., CA for 'name CA')
                    indices.append(selected_atoms[0].index)
                    # Store original residue ID
                    resids.append(residue.resid)
                    # Store chain ID (segment ID in MDAnalysis)
                    chain_ids.append(segment.segid if segment.segid else "A")

        return "".join(sequence), indices, resids, chain_ids

    def _perform_sequence_alignment(
        self, ref_seq: str, mob_seq: str
    ) -> Tuple[str, str, float]:
        """
        Perform pairwise sequence alignment.

        Args:
            ref_seq: Reference sequence
            mob_seq: Mobile sequence

        Returns:
            Tuple of (aligned_ref_seq, aligned_mob_seq, alignment_score)
        """
        alignments = self.aligner.align(ref_seq, mob_seq)
        best_alignment = alignments[0]

        return str(best_alignment[0]), str(best_alignment[1]), best_alignment.score

    def _get_matching_indices(
        self,
        aligned_ref: str,
        aligned_mob: str,
        ref_indices: List[int],
        mob_indices: List[int],
        ref_resids: List[int],
        mob_resids: List[int],
        ref_chain_ids: List[str],
        mob_chain_ids: List[str],
    ) -> Tuple[List[int], List[int], List[int], List[int], List[str], List[str]]:
        """
        Extract matching atom indices, residue IDs, and chain IDs from aligned sequences.

        Args:
            aligned_ref: Aligned reference sequence
            aligned_mob: Aligned mobile sequence
            ref_indices: Original reference atom indices
            mob_indices: Original mobile atom indices
            ref_resids: Original reference residue IDs
            mob_resids: Original mobile residue IDs
            ref_chain_ids: Original reference chain IDs
            mob_chain_ids: Original mobile chain IDs

        Returns:
            Tuple of (matched_ref_indices, matched_mob_indices, matched_ref_resids,
                     matched_mob_resids, matched_ref_chain_ids, matched_mob_chain_ids)
        """
        matched_ref = []
        matched_mob = []
        matched_ref_resids = []
        matched_mob_resids = []
        matched_ref_chain_ids = []
        matched_mob_chain_ids = []
        ref_pos = mob_pos = 0

        for ref_char, mob_char in zip(aligned_ref, aligned_mob):
            if ref_char != "-":
                ref_pos += 1
            if mob_char != "-":
                mob_pos += 1

            # Both positions have residues (no gaps)
            if ref_char != "-" and mob_char != "-":
                matched_ref.append(ref_indices[ref_pos - 1])
                matched_mob.append(mob_indices[mob_pos - 1])
                matched_ref_resids.append(ref_resids[ref_pos - 1])
                matched_mob_resids.append(mob_resids[mob_pos - 1])
                matched_ref_chain_ids.append(ref_chain_ids[ref_pos - 1])
                matched_mob_chain_ids.append(mob_chain_ids[mob_pos - 1])

        return (
            matched_ref,
            matched_mob,
            matched_ref_resids,
            matched_mob_resids,
            matched_ref_chain_ids,
            matched_mob_chain_ids,
        )

    def align(
        self, reference: mda.Universe, mobile: mda.Universe, selection: str = "name CA"
    ) -> AlignmentResult:
        """
        Perform structural alignment of two protein structures.

        Args:
            reference: Reference MDAnalysis Universe
            mobile: Mobile MDAnalysis Universe to be aligned
            selection: Atom selection string (default: "name CA")

        Returns:
            AlignmentResult object containing alignment details and results
        """
        # Extract sequences and indices
        ref_seq, ref_indices, ref_resids, ref_chain_ids = (
            self._extract_sequence_and_indices(reference, selection)
        )
        mob_seq, mob_indices, mob_resids, mob_chain_ids = (
            self._extract_sequence_and_indices(mobile, selection)
        )

        # Perform sequence alignment
        aligned_ref, aligned_mob, score = self._perform_sequence_alignment(
            ref_seq, mob_seq
        )
        if self.verbose:
            print(f"Sequence alignment score: {score}")

        # Get matching indices
        (
            matched_ref_indices,
            matched_mob_indices,
            matched_ref_resids,
            matched_mob_resids,
            matched_ref_chain_ids,
            matched_mob_chain_ids,
        ) = self._get_matching_indices(
            aligned_ref,
            aligned_mob,
            ref_indices,
            mob_indices,
            ref_resids,
            mob_resids,
            ref_chain_ids,
            mob_chain_ids,
        )
        if self.verbose:
            print(f"Matched reference residue IDs: {matched_ref_resids}")
            print(f"Matched mobile residue IDs: {matched_mob_resids}")
            print(f"Number of matched residues: {len(matched_ref_resids)}")

        # Create atom groups for structural alignment
        ref_atoms = reference.atoms[matched_ref_indices]
        mob_atoms = mobile.atoms[matched_mob_indices]

        # Perform structural alignment
        rmsd_before, rmsd_after = alignto(mob_atoms, ref_atoms, match_atoms=False)

        # Calculate per-residue distances after alignment
        # Refresh atom groups to get updated positions
        ref_atoms = reference.atoms[matched_ref_indices]
        mob_atoms = mobile.atoms[matched_mob_indices]
        position_distances = np.linalg.norm(
            ref_atoms.positions - mob_atoms.positions, axis=1
        ).tolist()

        # Create sequence alignment object
        seq_alignment = SequenceAlignment(
            reference_sequence=aligned_ref,
            mobile_sequence=aligned_mob,
            reference_indices=matched_ref_indices,
            mobile_indices=matched_mob_indices,
            reference_resids=matched_ref_resids,
            mobile_resids=matched_mob_resids,
            reference_chain_ids=matched_ref_chain_ids,
            mobile_chain_ids=matched_mob_chain_ids,
            alignment_score=score,
        )

        # Create and return complete alignment result
        return AlignmentResult(
            sequence_alignment=seq_alignment,
            rmsd_before=rmsd_before,
            rmsd_after=rmsd_after,
            n_aligned_residues=len(matched_ref_indices),
            position_distances=position_distances,
        )

    def calculate_position_distances(self, result: AlignmentResult) -> np.ndarray:
        """
        Calculate position-wise distances from alignment result.

        Args:
            result: AlignmentResult object

        Returns:
            Array of per-residue distances
        """
        return result.get_distances_array()

    def plot_distances(
        self,
        result: AlignmentResult,
        title: Optional[str] = None,
        xlabel: str = "Residue Position",
        ylabel: str = "Distance (Å)",
        figsize: Tuple[int, int] = (12, 8),
        show_chains: bool = True,
        chain_colors: Optional[Dict[str, str]] = None,
    ) -> plt.Figure:
        """
        Plot position-wise distances after alignment with chain information.

        Args:
            result: AlignmentResult object
            title: Plot title (default: auto-generated)
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size tuple
            show_chains: Whether to show chain information
            chain_colors: Custom colors for chains (dict: chain_id -> color)

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        distances = result.get_distances_array()
        ref_resids = result.sequence_alignment.reference_resids
        ref_chains = result.sequence_alignment.reference_chain_ids

        if show_chains:
            # Create chain-aware x-axis
            chain_boundaries = []
            unique_chains = []
            current_chain = None

            for i, chain in enumerate(ref_chains):
                if chain != current_chain:
                    if current_chain is not None:
                        chain_boundaries.append(i)
                    unique_chains.append(chain)
                    current_chain = chain

            # Default colors for chains
            if chain_colors is None:
                default_colors = [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                    "#8c564b",
                    "#e377c2",
                    "#7f7f7f",
                    "#bcbd22",
                    "#17becf",
                ]
                chain_colors = {
                    chain: default_colors[i % len(default_colors)]
                    for i, chain in enumerate(unique_chains)
                }

            # Plot with chain colors
            plotting_chain: Optional[str] = None
            start_idx = 0

            for i, chain in enumerate(
                ref_chains + [None]
            ):  # Add None to trigger final segment
                if chain != plotting_chain:
                    if plotting_chain is not None:
                        # Plot the segment
                        chain_color = chain_colors.get(plotting_chain, "#1f77b4")
                        x_vals = range(start_idx, i)
                        y_vals = distances[start_idx:i]
                        ax.plot(
                            x_vals,
                            y_vals,
                            "o-",
                            color=chain_color,
                            linewidth=2,
                            markersize=4,
                            alpha=0.7,
                            label=f"Chain {plotting_chain}",
                        )

                    start_idx = i
                    if chain is not None:
                        plotting_chain = chain

            # Create custom x-axis labels
            n_points = len(ref_resids)
            if n_points <= 50:
                # Show all residue IDs for small proteins
                ax.set_xticks(range(n_points))
                ax.set_xticklabels(
                    [
                        f"{resid}\n{chain}"
                        for resid, chain in zip(ref_resids, ref_chains)
                    ],
                    rotation=45,
                    ha="right",
                    fontsize=8,
                )
            else:
                # Show selected residue IDs for large proteins
                step = max(1, n_points // 20)
                tick_positions = range(0, n_points, step)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(
                    [f"{ref_resids[i]}\n{ref_chains[i]}" for i in tick_positions],
                    rotation=45,
                    ha="right",
                    fontsize=8,
                )

            # Add legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        else:
            # Simple plot without chain information
            ax.plot(distances, "b-", linewidth=2, alpha=0.7)
            ax.scatter(range(len(distances)), distances, c="red", s=30, alpha=0.6)

            # Use residue IDs on x-axis
            n_points = len(ref_resids)
            if n_points <= 50:
                ax.set_xticks(range(n_points))
                ax.set_xticklabels(
                    [str(resid) for resid in ref_resids], rotation=45, ha="right"
                )
            else:
                step = max(1, n_points // 20)
                tick_positions = range(0, n_points, step)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(
                    [str(ref_resids[i]) for i in tick_positions],
                    rotation=45,
                    ha="right",
                )

        if title is None:
            chain_info = (
                f" ({len(set(ref_chains))} chains)"
                if show_chains and len(set(ref_chains)) > 1
                else ""
            )
            title = f"Per-Residue Distances After Alignment{chain_info}\n(RMSD: {result.rmsd_after:.2f} Å, {result.n_aligned_residues} residues)"

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        # Add some statistics
        mean_dist = np.mean(distances)
        ax.axhline(
            y=mean_dist,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Mean: {mean_dist:.2f} Å",
        )

        plt.tight_layout()
        return fig

    def plot_distances_interactive(
        self,
        result: AlignmentResult,
        title: Optional[str] = None,
        show_chains: bool = True,
        chain_colors: Optional[Dict[str, str]] = None,
        height: int = 600,
        width: int = 1000,
    ) -> go.Figure:
        """
        Create an interactive Plotly plot of position-wise distances with rich hover information.

        Args:
            result: AlignmentResult object
            title: Plot title (default: auto-generated)
            show_chains: Whether to show chain information and colors
            chain_colors: Custom colors for chains (dict: chain_id -> color)
            height: Plot height in pixels
            width: Plot width in pixels

        Returns:
            Plotly Figure object with interactive features
        """
        distances = result.get_distances_array()
        ref_resids = result.sequence_alignment.reference_resids
        mob_resids = result.sequence_alignment.mobile_resids
        ref_chains = result.sequence_alignment.reference_chain_ids
        mob_chains = result.sequence_alignment.mobile_chain_ids
        ref_aas = list(result.sequence_alignment.reference_sequence)
        mob_aas = list(result.sequence_alignment.mobile_sequence)

        # Create DataFrame for easier plotting
        df = result.get_residue_info_table()

        if show_chains:
            # Get unique chains and assign colors
            unique_chains = df["ref_chain"].unique()

            if chain_colors is None:
                # Use Plotly's default color palette
                colors = px.colors.qualitative.Set1
                chain_colors = {
                    chain: colors[i % len(colors)]
                    for i, chain in enumerate(unique_chains)
                }

            # Create traces for each chain
            fig = go.Figure()

            for chain in unique_chains:
                chain_data = df[df["ref_chain"] == chain].reset_index()

                # Create hover text with rich information
                hover_text = []
                for _, row in chain_data.iterrows():
                    hover_info = (
                        f"<b>Residue Information</b><br>"
                        f"Reference: {row['ref_aa']}{row['ref_resid']} (Chain {row['ref_chain']})<br>"
                        f"Mobile: {row['mob_aa']}{row['mob_resid']} (Chain {row['mob_chain']})<br>"
                        f"Distance: {row['distance']:.2f} Å<br>"
                        f"Alignment Position: {chain_data.index[_]}"
                    )
                    hover_text.append(hover_info)

                # Add trace for this chain
                fig.add_trace(
                    go.Scatter(
                        x=chain_data.index,
                        y=chain_data["distance"],
                        mode="lines+markers",
                        name=f"Chain {chain}",
                        line=dict(color=chain_colors[chain], width=2),
                        marker=dict(size=6, color=chain_colors[chain]),
                        hovertemplate="%{hovertext}<extra></extra>",
                        hovertext=hover_text,
                        showlegend=True,
                    )
                )

            # Add chain boundary lines
            chain_boundaries = []
            current_pos = 0
            for chain in unique_chains[:-1]:  # Don't add line after last chain
                chain_length = len(df[df["ref_chain"] == chain])
                current_pos += chain_length
                chain_boundaries.append(current_pos - 0.5)

            for boundary in chain_boundaries:
                fig.add_vline(
                    x=boundary,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5,
                    annotation_text="Chain boundary",
                )

        else:
            # Simple plot without chain information
            hover_text = []
            for i in range(len(distances)):
                hover_info = (
                    f"<b>Residue Information</b><br>"
                    f"Reference: {ref_aas[i]}{ref_resids[i]} (Chain {ref_chains[i]})<br>"
                    f"Mobile: {mob_aas[i]}{mob_resids[i]} (Chain {mob_chains[i]})<br>"
                    f"Distance: {distances[i]:.2f} Å<br>"
                    f"Alignment Position: {i}"
                )
                hover_text.append(hover_info)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(distances))),
                    y=distances,
                    mode="lines+markers",
                    name="Residue Distances",
                    line=dict(color="blue", width=2),
                    marker=dict(size=6, color="red"),
                    hovertemplate="%{hovertext}<extra></extra>",
                    hovertext=hover_text,
                )
            )

        # Add mean distance line
        mean_dist = np.mean(distances)
        fig.add_hline(
            y=mean_dist,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Mean: {mean_dist:.2f} Å",
        )

        # Customize layout
        if title is None:
            chain_info = (
                f" ({len(set(ref_chains))} chains)"
                if show_chains and len(set(ref_chains)) > 1
                else ""
            )
            title = f"Interactive Residue Distance Analysis{chain_info}<br><sub>RMSD: {result.rmsd_after:.2f} Å, {result.n_aligned_residues} residues</sub>"

        # Create custom x-axis labels showing residue IDs
        n_points = len(ref_resids)
        if n_points <= 100:
            # Show more labels for smaller proteins
            step = max(1, n_points // 20)
        else:
            # Show fewer labels for larger proteins
            step = max(1, n_points // 15)

        tick_positions = list(range(0, n_points, step))
        tick_labels = [f"{ref_resids[i]}<br>{ref_chains[i]}" for i in tick_positions]

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis=dict(
                title="Residue Position (ID | Chain)",
                tickmode="array",
                tickvals=tick_positions,
                ticktext=tick_labels,
                tickangle=45,
            ),
            yaxis=dict(title="Distance (Å)"),
            hovermode="closest",
            showlegend=show_chains,
            height=height,
            width=width,
            template="plotly_white",
        )

        # Add range slider for large proteins
        if n_points > 50:
            fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type="linear"))

        return fig
