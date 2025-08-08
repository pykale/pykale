from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
from enum import Enum

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class SampleInfoMode(Enum):
    """Enumeration for sample information display modes."""
    NONE = "None"
    ALL = "All"
    AVERAGE = "Average"


@dataclass
class BoxplotStyleConfig:
    """Configuration class for boxplot styling with type hints and default values."""
    # Plot style
    matplotlib_style: str = "fivethirtyeight"
    hatch_type: str = "o"
    
    # Colors and styling
    box_edge_color: str = "black"
    box_linewidth: float = 1.0
    median_color: str = "crimson"
    median_linewidth: float = 3.0
    mean_facecolor: str = "crimson"
    mean_edgecolor: str = "black"
    mean_markersize: float = 10.0
    
    # Individual dots
    dot_alpha_normal: float = 0.75
    dot_alpha_alt: float = 0.2
    
    # Spacing values
    inner_spacing: float = 0.1
    middle_spacing: float = 0.02
    gap_large: float = 0.25
    gap_small: float = 0.12
    outer_gap_small: float = 0.35
    outer_gap_large: float = 0.24
    comparing_q_spacing: float = 0.2
    
    # Box widths
    default_box_width: float = 0.25
    comparing_q_width_base: float = 0.2
    
    # Font sizes
    sample_info_fontsize: int = 25
    legend_fontsize: int = 20
    
    # Layout
    legend_columnspacing: float = 2.0
    legend_bbox_anchor: Tuple[float, float] = (0.5, 1.18)
    subplot_bottom: float = 0.15
    subplot_left: float = 0.15
    
    # Figure saving
    figure_size: Tuple[float, float] = (16.0, 10.0)
    dpi: int = 600
    bbox_inches: str = "tight"
    pad_inches: float = 0.1

class BoxPlotter:
    """
    A comprehensive boxplot creator with flexible styling and configuration.
    
    This class provides methods to create various types of boxplots with
    customizable styling, sample information display, and legend handling.
    """
    
    def __init__(self, style_config: Optional[BoxplotStyleConfig] = None):
        """
        Initialize the BoxPlotter with optional style configuration.
        
        Args:
            style_config: Custom styling configuration. If None, uses default.
        """
        self.config = style_config or BoxplotStyleConfig()
        self.ax = None
        self.fig = None
        self.circ_patches: List[patches.Patch] = []
        self.max_bin_height = 0.0
        self.all_sample_label_x_locs: List[float] = []
        self.all_sample_percs: List[Union[float, List[float]]] = []  

    def setup_plot(self) -> None:
        """Initialize plot with common settings."""
        plt.style.use(self.config.matplotlib_style)
        self.fig, self.ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        self.ax.xaxis.grid(False)
