"""
Process experiment logs from LLM game-playing agents (COLM 2025, ICML 2026),
aggregate across trials, and generate per-game performance plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
from scipy import stats

# -- Plot style --
COLOR_RED = '#E64D3D'
COLOR_BLUE = '#3298DA'
COLOR_GREEN = '#25AE60'
COLOR_ORANGE = '#F39D16'
COLOR_PURPLE = '#9A5CB5'
COLOR_BLACK = '#020202'

TITLE_FONT_SIZE = 25
LABEL_FONT_SIZE = 20
LEGEND_FONT_SIZE = 14
TICK_FONT_SIZE = 22
STAR_SIZE = 22

plt.rcParams['font.size'] = TICK_FONT_SIZE
plt.rcParams['axes.titlesize'] = TITLE_FONT_SIZE
plt.rcParams['axes.labelsize'] = LABEL_FONT_SIZE
plt.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
plt.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
plt.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE

# -- Paths & experiment config --
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

EXPERIMENT_BATCHES = {
    'colm2025': {
        'folder': os.path.join(RESULTS_DIR, 'full_horizon_colm2025_logs'),
        'layout': 'colm',
    },
    'colm2025_horizon1': {
        'folder': os.path.join(RESULTS_DIR, 'one_step_horizon_colm2025_logs'),
        'layout': 'colm',
        'game_suffix': '_horizon1',
    },
    'icml2026': {
        'folder': os.path.join(RESULTS_DIR, 'full_horizon_icml2026_logs'),
        'layout': 'icml',
    },
}

GAME_STEP_CONFIG = {
    'breakout': {'threshold': 100, 'min_threshold': 30},
}
DEFAULT_STEP_CONFIG = {'threshold': 30, 'min_threshold': 20}

CI_CONFIDENCE = 0.66

# =========================================================================
# CSV Discovery
# =========================================================================

def discover_csvs_colm(folder):
    """COLM layout: {folder}/{game}/perf_*.csv"""
    game_csvs = {}
    if not os.path.isdir(folder):
        print(f"[WARN] Folder not found: {folder}")
        return game_csvs
    for game_dir in sorted(os.listdir(folder)):
        game_path = os.path.join(folder, game_dir)
        if not os.path.isdir(game_path):
            continue
        csvs = sorted(glob.glob(os.path.join(game_path, 'perf_*.csv')))
        if csvs:
            game_csvs[game_dir] = csvs
    return game_csvs


def discover_csvs_icml(folder):
    """ICML layout: {folder}/{game}_{YYYYMMDD_HHMMSS}/perf.csv"""
    game_csvs = {}
    if not os.path.isdir(folder):
        print(f"[WARN] Folder not found: {folder}")
        return game_csvs

    timestamp_re = re.compile(r'^(.+?)_(\d{8}_\d{6})$')
    for trial_dir in sorted(os.listdir(folder)):
        trial_path = os.path.join(folder, trial_dir)
        if not os.path.isdir(trial_path):
            continue
        perf_csv = os.path.join(trial_path, 'perf.csv')
        if not os.path.isfile(perf_csv):
            continue
        m = timestamp_re.match(trial_dir)
        game_name = m.group(1) if m else trial_dir
        game_csvs.setdefault(game_name, []).append(perf_csv)
    return game_csvs


def discover_all_csvs(batches=None):
    """Merge CSV paths from all experiment batches.

    Returns dict: game_name -> list of CSV paths.
    """
    if batches is None:
        batches = EXPERIMENT_BATCHES

    all_csvs = {}
    for batch_name, cfg in batches.items():
        folder = cfg['folder']
        layout = cfg['layout']

        if layout == 'colm':
            batch_csvs = discover_csvs_colm(folder)
        elif layout == 'icml':
            batch_csvs = discover_csvs_icml(folder)
        else:
            print(f"[WARN] Unknown layout '{layout}' for batch '{batch_name}'")
            continue

        game_suffix = cfg.get('game_suffix', '')
        for game_name, csv_list in batch_csvs.items():
            suffixed_name = game_name + game_suffix
            all_csvs.setdefault(suffixed_name, []).extend(csv_list)
            print(f"  [{batch_name}] {suffixed_name}: {len(csv_list)} trial(s)")

    return all_csvs


# =========================================================================
# CSV Loading
# =========================================================================

def load_trial_csv(file_path):
    """Load a single trial CSV, dropping any spurious index column."""
    df = pd.read_csv(file_path)

    # Some Pong CSVs have an extra unnamed index column
    if df.columns[0] == '' or df.columns[0].startswith('Unnamed'):
        df = df.drop(columns=[df.columns[0]])

    for col in ['Optimization Step', 'Mean Reward']:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {file_path}")

    return df


# =========================================================================
# Core Processing
# =========================================================================

def process_game_data(game_name, csv_files):
    """Aggregate trial CSVs for one game.

    Per trial: compute cumulative-best reward, forward-fill missing steps.
    Across trials: compute mean and CI per step.

    Returns (result_df, best_reward, best_step, max_step) or
            (None, None, None, None) if no files.
    """
    if not csv_files:
        print(f"No data files found for {game_name}")
        return None, None, None, None

    step_cfg = GAME_STEP_CONFIG.get(game_name, DEFAULT_STEP_CONFIG)
    threshold = step_cfg['threshold']
    min_threshold = step_cfg['min_threshold']

    # Load all files and find max step
    max_step_across_files = 0
    dfs = []
    for fp in csv_files:
        df = load_trial_csv(fp)
        df = df.fillna(0)
        dfs.append(df)
        max_step_across_files = max(max_step_across_files,
                                    int(df['Optimization Step'].max()))

    max_step = min(max_step_across_files, threshold)
    max_step = max(max_step, min_threshold)

    num_files = len(dfs)
    rewards_array = np.zeros((num_files, int(max_step) + 1))
    best_reward = float('-inf')
    best_step = None

    for file_idx, df in enumerate(dfs):
        best_reward_so_far = float('-inf')

        for _, row in df.iterrows():
            step = int(row['Optimization Step'])
            reward = row['Mean Reward']
            if step > max_step:
                continue
            if reward > best_reward_so_far:
                best_reward_so_far = reward
            rewards_array[file_idx, step] = best_reward_so_far
            if best_reward_so_far > best_reward:
                best_reward = best_reward_so_far
                best_step = step

        # Forward-fill gaps with last known best
        last_best_reward = 0
        last_valid_step = -1
        for step in range(int(max_step) + 1):
            if rewards_array[file_idx, step] > 0:
                last_valid_step = step
                last_best_reward = rewards_array[file_idx, step]
            elif last_valid_step != -1:
                rewards_array[file_idx, step] = last_best_reward

    max_step = min(max_step, threshold)

    print(f"\n{game_name} Statistics:")
    print(f"  Trials: {num_files}")
    print(f"  Max optimisation step (capped): {max_step}")
    print(f"  Best reward across all runs: {best_reward} at step {best_step}")

    # Mean and CI per step
    result_data = {'Step': [], 'Mean': [], 'CI_Lower': [], 'CI_Upper': [],
                   'Num_Trials': []}

    for step in range(int(max_step) + 1):
        step_rewards = rewards_array[:, step]
        mean_reward = np.mean(step_rewards)

        if len(step_rewards) > 1 and np.std(step_rewards) > 0:
            ci = stats.t.interval(CI_CONFIDENCE, len(step_rewards) - 1,
                                  loc=mean_reward,
                                  scale=stats.sem(step_rewards))
            ci_lower, ci_upper = ci
        else:
            ci_lower = mean_reward * 0.9
            ci_upper = mean_reward * 1.1

        result_data['Step'].append(step)
        result_data['Mean'].append(mean_reward)
        result_data['CI_Lower'].append(ci_lower)
        result_data['CI_Upper'].append(ci_upper)
        result_data['Num_Trials'].append(num_files)

    return pd.DataFrame(result_data), best_reward, best_step, max_step


# =========================================================================
# Save Aggregated Results
# =========================================================================

def save_aggregate_csvs(output_dir='results/aggregated'):
    """Process all games and save per-game + combined aggregate CSVs."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Discovering experiment CSVs...")
    print("=" * 60)
    all_csvs = discover_all_csvs()

    results = {}
    combined_frames = []

    print("\n" + "=" * 60)
    print("Processing games...")
    print("=" * 60)

    for game_name in sorted(all_csvs.keys()):
        csv_files = all_csvs[game_name]
        print(f"\n--- {game_name} ({len(csv_files)} trial(s)) ---")
        for fp in csv_files:
            print(f"  {fp}")

        result_df, best_reward, best_step, max_step = process_game_data(
            game_name, csv_files)

        if result_df is None:
            continue

        out_path = os.path.join(output_dir, f'{game_name}_aggregate.csv')
        result_df.to_csv(out_path, index=False)
        print(f"  -> Saved: {out_path}")

        results[game_name] = result_df

        combined_df = result_df.copy()
        combined_df.insert(0, 'Game', game_name)
        combined_frames.append(combined_df)

    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        combined_path = os.path.join(output_dir, 'all_games_aggregate.csv')
        combined.to_csv(combined_path, index=False)
        print(f"\n{'=' * 60}")
        print(f"Combined aggregate saved: {combined_path}")
        print(f"Games: {sorted(results.keys())}")
        print(f"{'=' * 60}")

    return results


# =========================================================================
# Plotting
# =========================================================================

def load_aggregate(game_name, aggregate_dir='results/aggregated'):
    """Load a saved aggregate CSV. Returns DataFrame or None."""
    path = os.path.join(aggregate_dir, f'{game_name}_aggregate.csv')
    if not os.path.isfile(path):
        print(f"[WARN] Aggregate not found: {path}")
        return None
    return pd.read_csv(path)


# (game_name, display_title, optional y-limits)
ALL_GAMES = [
    ('pong',            'Pong',             None),
    ('breakout',        'Breakout',         None),
    ('space_invaders',  'Space Invaders',   None),
    ('freeway',         'Freeway',          None),
    ('asterix',         'Asterix',          None),
    ('enduro',          'Enduro',           None),
    ('qbert',           'Q*bert',           None),
    ('seaquest',        'Seaquest',         None),
]


def plot_all_games(aggregate_dir='results/aggregated',
                   output_path='results/all_games_performance.pdf'):
    """2x4 grid of all games. Each subplot overlays horizon1 if available."""
    n_games = len(ALL_GAMES)
    nrows = 2
    ncols = 4

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    for idx, (game, title, ylim) in enumerate(ALL_GAMES):
        ax = axes[idx]

        df_full = load_aggregate(game, aggregate_dir)
        if df_full is not None:
            steps_full = df_full['Step'] + 1  # 1-indexed
            ax.plot(steps_full, df_full['Mean'],
                    color=COLOR_RED, linewidth=2, label='Multi-Step')
            ax.fill_between(steps_full, df_full['CI_Lower'], df_full['CI_Upper'],
                            color=COLOR_RED, alpha=0.15)

        df_h1 = load_aggregate(f'{game}_horizon1', aggregate_dir)
        if df_h1 is not None:
            steps_h1 = df_h1['Step'] + 1
            ax.plot(steps_h1, df_h1['Mean'],
                    color=COLOR_BLUE, linewidth=2, label='One-Step')
            ax.fill_between(steps_h1, df_h1['CI_Lower'], df_h1['CI_Upper'],
                            color=COLOR_BLUE, alpha=0.15)

        ax.set_xlabel('Optimization Step', fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel('Score', fontsize=LABEL_FONT_SIZE)
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.grid(False)
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

        # X-ticks: multiples of 5, skip 0, avoid crowding at the end
        max_step = int(max(
            df_full['Step'].max() + 1 if df_full is not None else 0,
            df_h1['Step'].max() + 1 if df_h1 is not None else 0,
        ))

        if max_step <= 0:
            xticks = []
        elif game == 'breakout' and max_step >= 100:
            xticks = [10, 30, 50, 70, 90]
        else:
            xticks = list(range(5, max_step + 1, 5))
            if max_step not in xticks:
                if not xticks or (max_step - xticks[-1]) >= 3:
                    xticks.append(max_step)

        ax.set_xticks(sorted(set(xticks)))

        if ylim is not None:
            ax.set_ylim(*ylim)

    for idx in range(n_games, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close()


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    results = save_aggregate_csvs()

    print("\n\nSUMMARY")
    print("-" * 40)
    for game_name, df in sorted(results.items()):
        final_mean = df['Mean'].iloc[-1]
        peak_mean = df['Mean'].max()
        n_trials = df['Num_Trials'].iloc[0]
        print(f"  {game_name:20s}  trials={n_trials}  "
              f"final_mean={final_mean:8.2f}  peak_mean={peak_mean:8.2f}")

    plot_all_games()
