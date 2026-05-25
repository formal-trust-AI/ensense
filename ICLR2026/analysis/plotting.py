import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup directories and files
BASE_DIR = Path(__file__).resolve().parent.parent
PAIRWISE_CSV = BASE_DIR / "run_test_pca_pairwise.csv"
PLOTS_DIR = BASE_DIR / "analysis" / "plots"

def load_data():
    if not PAIRWISE_CSV.exists():
        print(f"Error: Could not find {PAIRWISE_CSV}")
        return None
    
    df = pd.read_csv(PAIRWISE_CSV)
    print(f"Loaded pairwise data: {df.shape[0]} rows.")
    return df

def plot_win_matrix(df, metric, title, filename):
    """
    Plots a heatmap showing the win percentage of the row variant vs the column variant.
    metric is the column that dictates the winner (e.g., 'winner_pair_l2')
    """
    # Get unique variants
    variants = pd.unique(df[['lhs_variant', 'rhs_variant']].values.ravel('K'))
    variants = [v for v in variants if pd.notna(v) and v != 'tie']
    
    # Initialize win matrix
    win_matrix = pd.DataFrame(0.0, index=variants, columns=variants)
    
    for v1 in variants:
        for v2 in variants:
            if v1 == v2:
                win_matrix.loc[v1, v2] = np.nan
                continue
            
            # Cases where lhs=v1 and rhs=v2
            mask1 = (df['lhs_variant'] == v1) & (df['rhs_variant'] == v2)
            wins1 = (df[mask1][metric] == v1).sum()
            total1 = mask1.sum()
            
            # Cases where lhs=v2 and rhs=v1
            mask2 = (df['lhs_variant'] == v2) & (df['rhs_variant'] == v1)
            wins2 = (df[mask2][metric] == v1).sum()
            total2 = mask2.sum()
            
            total_matches = total1 + total2
            if total_matches > 0:
                win_matrix.loc[v1, v2] = (wins1 + wins2) / total_matches
            else:
                win_matrix.loc[v1, v2] = np.nan

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    cax = ax.imshow(win_matrix.values, cmap="RdYlGn", vmin=0, vmax=1)
    
    # Add annotations
    for i in range(len(variants)):
        for j in range(len(variants)):
            val = win_matrix.values[i, j]
            if not np.isnan(val):
                text_color = "white" if (val < 0.2 or val > 0.8) else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", color=text_color)
                
    ax.set_xticks(np.arange(len(variants)))
    ax.set_yticks(np.arange(len(variants)))
    ax.set_xticklabels(variants, rotation=45, ha="right")
    ax.set_yticklabels(variants)
    
    plt.colorbar(cax, label='Win Rate (Row vs Col)')
    plt.title(title)
    plt.ylabel("Variant (Row)")
    plt.xlabel("Opponent (Column)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_overall_win_rate(df, metric, title, filename):
    """
    Plots the overall win percentage of each variant across all its matchups.
    """
    variants = pd.unique(df[['lhs_variant', 'rhs_variant']].values.ravel('K'))
    variants = [v for v in variants if pd.notna(v) and v != 'tie']
    
    win_rates = {}
    for v in variants:
        # Matchups involving v
        mask = (df['lhs_variant'] == v) | (df['rhs_variant'] == v)
        total_matches = mask.sum()
        wins = (df[mask][metric] == v).sum()
        
        if total_matches > 0:
            win_rates[v] = wins / total_matches
            
    # Sort and plot
    win_df = pd.DataFrame(list(win_rates.items()), columns=['Variant', 'Win Rate']).sort_values('Win Rate', ascending=False)
    
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(win_df))
    plt.barh(y_pos, win_df['Win Rate'], align='center', color='teal')
    plt.yticks(y_pos, win_df['Variant'])
    plt.gca().invert_yaxis()  # labels read top-to-bottom
    plt.title(title)
    plt.xlabel("Overall Win Percentage")
    plt.ylabel("Variant")
    plt.xlim(0, 1)
    
    # Add percentage labels
    for i, v in enumerate(win_df['Win Rate']):
        plt.text(v + 0.01, i, f"{v:.1%}", va='center')
        
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_average_value_difference(df, lhs_metric, rhs_metric, title, filename):
    """
    Plots the average difference between variants (e.g. how much smaller/larger is the L2 norm on average)
    """
    variants = pd.unique(df[['lhs_variant', 'rhs_variant']].values.ravel('K'))
    variants = [v for v in variants if pd.notna(v) and v != 'tie']
    
    diff_matrix = pd.DataFrame(0.0, index=variants, columns=variants)
    
    for v1 in variants:
        for v2 in variants:
            if v1 == v2:
                diff_matrix.loc[v1, v2] = np.nan
                continue
            
            mask1 = (df['lhs_variant'] == v1) & (df['rhs_variant'] == v2)
            diff1 = (df[mask1][lhs_metric] - df[mask1][rhs_metric]).sum()
            
            mask2 = (df['lhs_variant'] == v2) & (df['rhs_variant'] == v1)
            diff2 = (df[mask2][rhs_metric] - df[mask2][lhs_metric]).sum()
            
            total = mask1.sum() + mask2.sum()
            
            if total > 0:
                diff_matrix.loc[v1, v2] = (diff1 + diff2) / total
            else:
                diff_matrix.loc[v1, v2] = np.nan
                
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    max_val = np.nanmax(np.abs(diff_matrix.values))
    if np.isnan(max_val) or max_val == 0:
        max_val = 1.0 # fallback
        
    cax = ax.imshow(diff_matrix.values, cmap="coolwarm", vmin=-max_val, vmax=max_val)
    
    # Add annotations
    for i in range(len(variants)):
        for j in range(len(variants)):
            val = diff_matrix.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="black")

    ax.set_xticks(np.arange(len(variants)))
    ax.set_yticks(np.arange(len(variants)))
    ax.set_xticklabels(variants, rotation=45, ha="right")
    ax.set_yticklabels(variants)
    
    plt.colorbar(cax, label=f'Avg Difference (Row - Col)')
    plt.title(title)
    plt.ylabel("Variant (Row)")
    plt.xlabel("Opponent (Column)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_dataset_breakdown(df, winner_metric, title, filename):
    """
    Plots a stacked bar chart showing which variant wins most frequently per dataset.
    """
    # Count wins per dataset per variant
    win_counts = df.groupby(['dataset', winner_metric]).size().unstack(fill_value=0)
    
    # Normalize to get percentages
    win_pct = win_counts.div(win_counts.sum(axis=1), axis=0) * 100
    
    win_pct.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
    plt.title(title)
    plt.xlabel("Dataset")
    plt.ylabel("Win Percentage (%)")
    plt.legend(title="Winning Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename)
    plt.close()
    print(f"Saved: {filename}")

def export_dataset_win_rates(df):
    """
    Exports win rates for L1 and L2 metrics per dataset into separate text files.
    """
    dataset_dir = BASE_DIR / "analysis" / "dataset_win_rates"
    os.makedirs(dataset_dir, exist_ok=True)
    
    datasets = df['dataset'].unique()
    metrics = {
        'winner_pair_l2': 'Pairwise L2 Norm',
        'winner_mean_centroid_l1': 'Mean Centroid L1',
        'winner_mean_centroid_l2': 'Mean Centroid L2'
    }
    
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        if dataset_df.empty:
            continue
            
        variants = pd.unique(dataset_df[['lhs_variant', 'rhs_variant']].values.ravel('K'))
        variants = [v for v in variants if pd.notna(v) and v != 'tie']
        
        output_file = dataset_dir / f"{dataset}_win_rates.txt"
        
        all_metric_rates = {}
        with open(output_file, 'w') as f:
            f.write(f"Win Rates for Dataset: {dataset}\n")
            f.write("="*40 + "\n\n")
            
            for metric_col, metric_name in metrics.items():
                if metric_col not in dataset_df.columns:
                    continue
                    
                f.write(f"--- {metric_name} Win Rates ---\n")
                
                win_rates = {}
                for v in variants:
                    mask = (dataset_df['lhs_variant'] == v) | (dataset_df['rhs_variant'] == v)
                    total_matches = mask.sum()
                    wins = (dataset_df[mask][metric_col] == v).sum()
                    
                    if total_matches > 0:
                        win_rates[v] = (wins / total_matches) * 100
                        
                # Sort descending for text file
                win_rates_sorted = dict(sorted(win_rates.items(), key=lambda item: item[1], reverse=True))
                all_metric_rates[metric_name] = win_rates # keep unsorted for aligned plotting
                
                for variant, rate in win_rates_sorted.items():
                    f.write(f"{variant}: {rate:.1f}%\n")
                f.write("\n")
                
        print(f"Saved dataset win rates: {output_file.name}")
        
        if all_metric_rates:
            fig, ax = plt.subplots(figsize=(14, 7))
            x = np.arange(len(variants))
            width = 0.25
            
            for i, (metric_name, win_rates) in enumerate(all_metric_rates.items()):
                rates = [win_rates.get(v, 0) for v in variants]
                ax.bar(x + i*width - width, rates, width, label=metric_name)
                
            ax.set_ylabel('Win Percentage (%)')
            ax.set_title(f'Win Rates by Metric for Dataset: {dataset}')
            ax.set_xticks(x)
            ax.set_xticklabels(variants, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            hist_file = dataset_dir / f"{dataset}_win_rates_histogram.png"
            plt.savefig(hist_file)
            plt.close()
            print(f"Saved dataset histogram: {hist_file.name}")

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    df = load_data()
    if df is None or df.empty:
        print("No valid data found to plot.")
        return

    print(f"\nGenerating plots in {PLOTS_DIR} ...")
    
    # Analysis 1: Pairwise L2 Norm (Distance between generated sensitive samples)
    if 'winner_pair_l2' in df.columns:
        plot_win_matrix(df, 'winner_pair_l2', "Pairwise L2 Norm Win Matrix (Smaller is Winner)", "pair_l2_win_matrix.png")
        plot_overall_win_rate(df, 'winner_pair_l2', "Overall Win Rate for Pairwise L2 Norm", "pair_l2_overall_win_rate.png")
        plot_average_value_difference(df, 'lhs_pair_l2', 'rhs_pair_l2', "Average Difference in Pairwise L2 Norm (Row - Col)", "pair_l2_avg_diff.png")
        plot_dataset_breakdown(df, 'winner_pair_l2', "Pairwise L2 Norm Win Breakdown by Dataset", "pair_l2_dataset_breakdown.png")
        
    # Analysis 2: Mean Centroid L2 (Distance from centroid)
    if 'winner_mean_centroid_l2' in df.columns:
        plot_win_matrix(df, 'winner_mean_centroid_l2', "Mean Centroid L2 Win Matrix (Smaller is Winner)", "centroid_l2_win_matrix.png")
        plot_overall_win_rate(df, 'winner_mean_centroid_l2', "Overall Win Rate for Mean Centroid L2", "centroid_l2_overall_win_rate.png")
        plot_average_value_difference(df, 'lhs_mean_centroid_l2', 'rhs_mean_centroid_l2', "Average Difference in Mean Centroid L2 (Row - Col)", "centroid_l2_avg_diff.png")

    # Analysis 3: Mean Centroid L1
    if 'winner_mean_centroid_l1' in df.columns:
        plot_win_matrix(df, 'winner_mean_centroid_l1', "Mean Centroid L1 Win Matrix (Smaller is Winner)", "centroid_l1_win_matrix.png")
        plot_overall_win_rate(df, 'winner_mean_centroid_l1', "Overall Win Rate for Mean Centroid L1", "centroid_l1_overall_win_rate.png")
        plot_average_value_difference(df, 'lhs_mean_centroid_l1', 'rhs_mean_centroid_l1', "Average Difference in Mean Centroid L1 (Row - Col)", "centroid_l1_avg_diff.png")
        plot_dataset_breakdown(df, 'winner_mean_centroid_l1', "Mean Centroid L1 Win Breakdown by Dataset", "centroid_l1_dataset_breakdown.png")

    print("\nExporting dataset-specific win rates...")
    export_dataset_win_rates(df)

    print("\nAll analysis complete!")

if __name__ == "__main__":
    main()
