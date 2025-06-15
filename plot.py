import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

_shared_start_idx = None  # Global variable to share the start index

def get_time_range_str(sample_data):
    """Helper function to format time range string"""
    start_time = f"{int(sample_data.iloc[0]['Month'])}/{int(sample_data.iloc[0]['Day'])} {int(sample_data.iloc[0]['Hour']):02d}:00"
    end_time = f"{int(sample_data.iloc[-1]['Month'])}/{int(sample_data.iloc[-1]['Day'])} {int(sample_data.iloc[-1]['Hour']):02d}:00"
    return f"{start_time} to {end_time}"

def plot_160h_analysis(test_results_path="test_results.csv"):
    global _shared_start_idx
    df = pd.read_csv(test_results_path)
    
    if _shared_start_idx is None:
        _shared_start_idx = np.random.randint(0, len(df) - 160)
    start_idx = _shared_start_idx
    
    sample_data = df.iloc[start_idx:start_idx+160].copy().reset_index(drop=True)
    time_range = get_time_range_str(sample_data)
    
    plt.figure(figsize=(22, 10))
    hours = np.arange(len(sample_data))
    
    # Plot price curves
    plt.plot(hours, sample_data['Grid_Price'], color='#1f77b4', label='Grid Price', linewidth=1.5, alpha=0.8)
    plt.plot(hours, sample_data['Aggregator_Price'], color='#2ca02c', label='Aggregator Price', linewidth=1.5, alpha=0.8)
    
    # Action configuration with price-based positioning
    action_config = {
        0: {'color': '#d62728', 'label': 'Charge', 'marker': '^', 'price_col': 'Aggregator_Price'},
        1: {'color': '#008000', 'label': 'Discharge', 'marker': 'v', 'price_col': 'Grid_Price'},
        2: {'color': '#7f7f7f', 'label': 'Idle', 'marker': 'o', 'price_col': 'Grid_Price'}
    }
    
    # Plot actions at their actual price points
    for action in [0, 1, 2]:
        mask = sample_data['Action'] == action
        action_prices = sample_data.loc[mask, action_config[action]['price_col']]
        
        plt.scatter(
            hours[mask],
            action_prices,
            color=action_config[action]['color'],
            s=150,
            marker=action_config[action]['marker'],
            edgecolor='black',
            linewidth=1,
            alpha=0.9,
            zorder=3
        )
    
    # Add car availability shading
    car_available = sample_data['Car_Available'].astype(bool)
    for i in range(len(car_available)):
        if not car_available[i]:
            plt.axvspan(i-0.5, i+0.5, color='yellow', alpha=0.15, zorder=0)
    
    # Formatting
    plt.xlabel('Consecutive Hours (20-hour intervals)', fontsize=24)
    plt.ylabel('Price ($/kWh)', fontsize=24)
    plt.title(f'160-Hour Analysis: Actions & Pricing\n(Hours {start_idx}-{start_idx+160}: {time_range})', 
             fontsize=24, pad=20)
    
    # X-axis ticks every 20 hours
    plt.xticks(np.arange(0, 160, 20), [f'Hour {i}' for i in range(0, 160, 20)], fontsize=20)
    plt.yticks(fontsize=10)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', label='Grid Price'),
        Line2D([0], [0], color='#2ca02c', label='Aggregator Price'),
        Line2D([0], [0], marker='^', color='w', label='Charge @ Aggregator Price',
              markerfacecolor='#d62728', markersize=24, markeredgecolor='black'),
        Line2D([0], [0], marker='v', color='w', label='Discharge @ Grid Price',
              markerfacecolor='#008000', markersize=24, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='Idle @ Grid Price',
              markerfacecolor='#7f7f7f', markersize=24, markeredgecolor='black'),
        Patch(facecolor='yellow', alpha=0.15, label='Car Unavailable')
    ]
    
    plt.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=20)
    
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('action_price_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_soc(test_results_path="test_results.csv"):
    global _shared_start_idx
    df = pd.read_csv(test_results_path)
    
    if _shared_start_idx is None:
        _shared_start_idx = np.random.randint(0, len(df) - 160)
    start_idx = _shared_start_idx
    
    sample_data = df.iloc[start_idx:start_idx+160].copy()
    time_range = get_time_range_str(sample_data)
    
    plt.figure(figsize=(22, 8))
    
    # Plot SOC bars for available hours
    available_mask = sample_data['Car_Available'] == 1
    plt.bar(np.where(available_mask)[0], 
           sample_data[available_mask]['Final_SoC'] * 100,
           color='#1f77b4',
           alpha=0.7,
           width=0.8,
           label='SOC During Availability')
    
    # Shade unavailable hours
    for i in range(len(sample_data)):
        if sample_data.iloc[i]['Car_Available'] == 0:
            plt.axvspan(i-0.5, i+0.5, color='yellow', alpha=0.3, zorder=0)
    
    # Formatting
    plt.xlabel('Consecutive Hours in Selected Period', fontsize=24)
    plt.ylabel('State of Charge (%)', fontsize=24)
    plt.title(f'160-Hour Analysis: Battery State of Charge\n(Hours {start_idx}-{start_idx+160}: {time_range})', 
             fontsize=24, pad=20)
    
    # Axis configuration
    plt.ylim(0, 100)
    plt.xlim(-0.5, 159.5)
    plt.xticks(np.arange(0, 160, 20), 
              [f'Hour {i}' for i in range(0, 160, 20)], 
              fontsize=20)
    
    # Create custom legend
    legend_elements = [
        Patch(facecolor='#1f77b4', label='SOC During Availability'),
        Patch(facecolor='yellow', alpha=0.3, label='Car Unavailable')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('soc_timeline.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_160h_analysis()
plot_soc()