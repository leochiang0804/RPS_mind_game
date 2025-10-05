#!/usr/bin/env python3
"""
Simple script to view the generated plots
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

def view_latest_plots():
    """View the most recent visualization plots"""
    
    # Look for the most recent plots in visualizations folder
    plot_patterns = [
        'visualizations/win_rate_analysis_*.png',
        'visualizations/comprehensive_analysis_*.png',
        'visualizations/interaction_heatmaps_*.png'
    ]
    
    for pattern in plot_patterns:
        files = glob.glob(pattern)
        if files:
            # Get the most recent file
            latest_file = max(files, key=os.path.getctime)
            print(f"📊 Displaying: {latest_file}")
            
            # Load and display the plot
            img = mpimg.imread(latest_file)
            plt.figure(figsize=(15, 10))
            plt.imshow(img)
            plt.axis('off')
            
            # Set title based on file type
            if 'win_rate_analysis' in latest_file:
                title = '42-Opponent Win Rate Analysis - Box Plots'
            elif 'comprehensive_analysis' in latest_file:
                title = 'Comprehensive Performance Analysis'
            elif 'interaction_heatmaps' in latest_file:
                title = 'Factor Interaction Heatmaps'
            else:
                title = 'Opponent Analysis Visualization'
                
            plt.title(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            print(f"� {title} displayed! Close the window to continue to next plot.")
        else:
            print(f"⚠️ No files found matching pattern: {pattern}")

if __name__ == "__main__":
    view_latest_plots()