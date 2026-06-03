import os
import glob
import pandas as pd
import ast
import re
from scipy.stats import wilcoxon


BASE_DIR   = r"C:\Users\user\Desktop\MSML\msde"
MSDE_DIR   = r"C:\Users\user\Desktop\MSML\msde\adbench\result\MSDE results"
BENCH_DIR  = r"C:\Users\user\Desktop\MSML\msde\adbench\result\benchmarks results"


def clean_numpy(text):
    if isinstance(text, str):
        text = re.sub(r"np\.\w+\((.*?)\)", r"\1", text)
    return text


def parse_metrics(m):
    try:
        cleaned = clean_numpy(m)
        d = ast.literal_eval(cleaned)
        return pd.Series({
            'aucroc':     d.get('aucroc'),
            'aucpr':      d.get('aucpr'),
            'p_at_n':     d.get('p_at_n'),
            'adj_p_at_n': d.get('adj_p_at_n'),
            'adj_ap':     d.get('adj_ap'),
        })
    except Exception:
        return pd.Series({k: None for k in
                          ['aucroc', 'aucpr', 'p_at_n', 'adj_p_at_n', 'adj_ap']})


def parse_config(cfg):
    try:
        cleaned = clean_numpy(cfg)
        dataset, noise, seed = ast.literal_eval(cleaned)
        return dataset, noise, seed
    except Exception:
        return (None, None, None)



def load_csv(path):
    df = pd.read_csv(path, header=None, skiprows=1)
    df.columns = ['config', 'method', 'metrics', 'fit_time', 'inference_time']

    df[['dataset', 'noise', 'seed']] = (
        df['config'].apply(parse_config).apply(pd.Series)
    )
    df[['aucroc', 'aucpr', 'p_at_n', 'adj_p_at_n', 'adj_ap']] = (
        df['metrics'].apply(parse_metrics)
    )
    return df



def parse_experiment(exp_id):

    # ── MSDE ───────────────────────────────────────────────
    msde_pattern = os.path.join(MSDE_DIR, f"*{exp_id}*.csv")
    msde_files   = glob.glob(msde_pattern)

    if not msde_files:
        print(f"  [WARNING] No MSDE file found for experiment {exp_id} in:\n  {MSDE_DIR}")
        df_msml = pd.DataFrame()
    else:
        df_msml = pd.concat([load_csv(f) for f in msde_files], ignore_index=True)
        df_msml['method']     = 'MSDE'
        df_msml['experiment'] = exp_id

    # ── Benchmarks ─────────────────────────────────────────
    bench_pattern = os.path.join(BENCH_DIR, f"*{exp_id}*.csv")
    bench_files   = glob.glob(bench_pattern)

    if not bench_files:
        print(f"  [WARNING] No benchmark file found for experiment {exp_id} in:\n  {BENCH_DIR}")
        df_bench = pd.DataFrame()
    else:
        df_bench = pd.concat([load_csv(f) for f in bench_files], ignore_index=True)
        df_bench = df_bench[df_bench['method'] != 'DeepSVDD']   # exclude DeepSVDD
        df_bench['experiment'] = exp_id

    return pd.concat([df_msml, df_bench], ignore_index=True)



experiment_ids = [16, 17, 18, 19]

all_experiments = []

for exp in experiment_ids:
    print(f"Loading Experiment {exp} ...")
    df_exp = parse_experiment(exp)
    all_experiments.append(df_exp)

df_all = pd.concat(all_experiments, ignore_index=True)

print("\nFinal Shape:", df_all.shape)
print("Methods found:", df_all['method'].unique().tolist())



pivot_df = df_all.pivot_table(
    index=['dataset', 'noise', 'seed', 'experiment'],
    columns='method',
    values='p_at_n',
    aggfunc='mean',   # handles rare duplicates gracefully instead of erroring
)

print("\nAvailable Methods:")
print(list(pivot_df.columns))



methods = [m for m in pivot_df.columns if m != 'MSDE']

results = []

for method in methods:

    print(f"\nTesting MSDE vs {method}")

    paired = pivot_df[['MSDE', method]].dropna()
    n = len(paired)
    print(f"  Paired samples: {n}")

    if n < 10:
        # Wilcoxon needs enough pairs to be meaningful;
        # scipy itself requires >= 1 non-zero difference.
        print("  Skipped (too few samples — need at least 10 for a reliable test)")
        continue

    # Check for zero differences (all identical → test is undefined)
    diffs = paired['MSDE'] - paired[method]
    if (diffs == 0).all():
        print("  Skipped (all differences are zero — methods produce identical scores)")
        continue

    stat, p = wilcoxon(paired['MSDE'], paired[method], alternative='greater')

    results.append({
        'Comparison':          f'MSDE vs {method}',
        'Num Samples':         n,
        'MSDE Mean Precision@n':    paired['MSDE'].mean(),
        f'{method} Mean Precision@n': paired[method].mean(),
        'Mean Difference':     diffs.mean(),
        'Statistic':           stat,
        'P-value':             p,
        'Significant (<0.05)': p < 0.05,
    })


# ==========================================================
# 11. Display & Save Results
# ==========================================================
results_df = pd.DataFrame(results).sort_values('P-value').reset_index(drop=True)

print("\n" + "=" * 60)
print("Wilcoxon Signed-Rank Test Results  (metric: Precision@n)")
print("=" * 60 + "\n")
print(results_df.to_string(index=False))

out_path = os.path.join(BASE_DIR, "wilcoxon_results_p_at_n.csv")
results_df.to_csv(out_path, index=False)
print(f"\nResults saved to: {out_path}")
