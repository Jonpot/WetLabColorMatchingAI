import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# ─── Helpers ────────────────────────────────────────────────────────────────

def load_plate_data(csv_path):
    df = pd.read_csv(csv_path)
    # mark rotated vs original
    df['rotated'] = df['well'].str.endswith("'")
    # canonical well name (drop the apostrophe)
    df['well0'] = df['well'].str.rstrip("'")
    return df

def compute_fractions(df):
    # total volume is 200 uL
    for dye in ['red','yellow','green','water']:
        df[f'{dye}_frac'] = df[f'{dye}_vol'] / 200
    return df

def split_experiments(df):
    # Linear tests: exactly one dye has nonzero volume and that is a multiple of 20
    dyes = ['red','yellow','green']
    is_linear = df.apply(
        lambda r: sum(r[f'{d}_vol']>0 for d in dyes)==1
                  and r['water_vol']>=0, axis=1
    )
    return df[is_linear].copy(), df[~is_linear].copy()

# ─── Linear‐space analysis ─────────────────────────────────────────────────

def analyze_linear(df_lin, outdir='linear_results'):
    Path(outdir).mkdir(exist_ok=True)
    results = {}
    for dye in ['red','yellow','green']:
        # select rows where that dye is the only non‐zero
        sel = df_lin[df_lin[f'{dye}_vol']>0]
        if sel.empty:
            continue

        X = sel[[f'{dye}_frac']].values  # shape (n,1)
        for channel, meas in [('red','measured_red'),
                              ('green','measured_green'),
                              ('blue','measured_blue')]:
            y = sel[meas].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            results[(dye,channel)] = {'slope': model.coef_[0],
                                     'intercept': model.intercept_,
                                     'rmse': rmse}

            # plot
            plt.figure(figsize=(5,4))
            plt.scatter(X, y, label='data')
            xs = np.linspace(0,1,101).reshape(-1,1)
            plt.plot(xs, model.predict(xs), '-', label='fit')
            plt.title(f'{dye.title()} dye → {channel} channel\n'
                      f'slope={model.coef_[0]:.1f}, RMSE={rmse:.1f}')
            plt.xlabel(f'{dye} fraction')
            plt.ylabel(f'{channel} intensity')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{outdir}/{dye}_{channel}.png')
            plt.close()

    return pd.DataFrame.from_dict(results, orient='index')

# ─── Mixture analysis ──────────────────────────────────────────────────────

def analyze_mixtures(df_mix, lin_fits, outdir='mix_results'):
    Path(outdir).mkdir(exist_ok=True)
    # Build additive predictions: predicted_intensity = sum(frac_dye * (slope*1 + intercept*0?))
    # Actually: pred = intercept + slope * frac_dye for each dye, sum across dyes?
    # More correct: each dye contributes slope_i * frac_i + intercept_i * frac_i
    # But intercept should be attributed to water baseline: we’ll assume water channel ~ background.
    # Simpler: pred = sum(frac_dye * slope_i * 200) + water_frac * background
    # Here we'll do: predicted = sum(frac_dye * slope_i + frac_dye*intercept_i)
    # (rough but illustrative)

    mix_metrics = {}
    for channel in ['red','green','blue']:
        y_true = df_mix[f'measured_{channel}'].values
        y_pred = np.zeros_like(y_true, dtype=float)
        for dye in ['red','yellow','green']:
            fit = lin_fits.loc[(dye,channel)]
            y_pred += df_mix[f'{dye}_frac'] * fit['slope'] + df_mix[f'{dye}_frac']*fit['intercept']
        # optionally add water term if you have it
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mix_metrics[channel] = {'rmse': rmse}
        # scatter plot
        plt.figure(figsize=(5,5))
        plt.scatter(y_pred, y_true, alpha=0.6)
        mx = max(y_true.max(), y_pred.max())
        plt.plot([0,mx],[0,mx],'k--')
        plt.xlabel('Predicted intensity')
        plt.ylabel('Measured intensity')
        plt.title(f'Mixture: {channel} channel RMSE={rmse:.1f}')
        plt.tight_layout()
        plt.savefig(f'{outdir}/mix_{channel}.png')
        plt.close()

    return pd.DataFrame.from_dict(mix_metrics, orient='index')

def analyze_mixtures_with_water(df_mix, lin_fits, df_all, outdir='mix_results'):
    """
    df_mix: mixture wells
    lin_fits: slopes only (we’ll drop intercepts here)
    df_all: full DataFrame (so we can find pure-water wells)
    """
    Path(outdir).mkdir(exist_ok=True)

    # 1) find water-only baseline per channel
    water_wells = df_all[
        (df_all.red_vol   == 0) &
        (df_all.yellow_vol== 0) &
        (df_all.green_vol == 0)
    ]
    baseline = {
        ch: water_wells[f'measured_{ch}'].mean()
        for ch in ['red','green','blue']
    }

    mix_metrics = {}
    for ch in ['red','green','blue']:
        y_true = df_mix[f'measured_{ch}'].values
        # 2) build predicted:
        #   water_frac * baseline  +  sum(dye_frac * slope_dye→ch)
        y_pred = (
            df_mix['water_frac'] * baseline[ch]
            + sum(
                df_mix[f'{dye}_frac'] * lin_fits.loc[(dye,ch),'slope']
                for dye in ['red','yellow','green']
            )
        )

        # 3) RMSE + scatter
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mix_metrics[ch] = {'rmse': rmse}

        plt.figure(figsize=(5,5))
        plt.scatter(y_pred, y_true, alpha=0.6)
        m = max(y_true.max(), y_pred.max())
        plt.plot([0,m],[0,m],'k--', linewidth=1)
        plt.title(f'{ch.title()} channel RMSE={rmse:.1f}\n(includes water baseline)')
        plt.xlabel('Predicted intensity')
        plt.ylabel('Measured intensity')
        plt.tight_layout()
        plt.savefig(f'{outdir}/mix_{ch}_with_water.png')
        plt.close()

    return pd.DataFrame.from_dict(mix_metrics, orient='index')

def analyze_mixtures_with_intercept(df_mix, lin_fits, df_all, outdir='mix_results'):
    """
    Predict each channel as:
      I_pred = baseline + sum(frac_dye * slope_dye→channel)
    where baseline = avg pure-water reading per channel.
    """
    Path(outdir).mkdir(exist_ok=True)

    # 1) estimate single baseline per channel
    water = df_all[
        (df_all.red_vol    == 0) &
        (df_all.yellow_vol == 0) &
        (df_all.green_vol  == 0)
    ]
    baseline = {ch: water[f'measured_{ch}'].mean()
                for ch in ['red','green','blue']}

    metrics = {}
    for ch in ['red','green','blue']:
        y_true = df_mix[f'measured_{ch}'].values

        # 2) build prediction = intercept + sum(frac*slope)
        y_pred = baseline[ch] + sum(
            df_mix[f'{dye}_frac'] * lin_fits.loc[(dye,ch),'slope']
            for dye in ['red','yellow','green']
        )

        # 3) compute RMSE + plot
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics[ch] = {'rmse': rmse}

        plt.figure(figsize=(5,5))
        plt.scatter(y_pred, y_true, alpha=0.6)
        m = max(y_true.max(), y_pred.max())
        plt.plot([0,m],[0,m],'k--',linewidth=1)
        plt.title(f'{ch.title()} channel\nRMSE={rmse:.1f}')
        plt.xlabel('Predicted intensity')
        plt.ylabel('Measured intensity')
        plt.tight_layout()
        plt.savefig(f'{outdir}/mix_{ch}_intercept.png')
        plt.close()

    return pd.DataFrame.from_dict(metrics, orient='index')


# ─── Extras: PCA ────────────────────────────────────────────────────────────

def analyze_pca(df_all, outdir='pca'):
    Path(outdir).mkdir(exist_ok=True)
    channels = ['measured_red','measured_green','measured_blue']
    pca = PCA(n_components=2)
    coords = pca.fit_transform(df_all[channels])
    plt.figure(figsize=(6,5))
    plt.scatter(coords[:,0], coords[:,1], c=coords[:,0], cmap='Spectral', alpha=0.7)
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.title('PCA of all wells')
    plt.tight_layout()
    plt.savefig(f'{outdir}/pca_all.png')
    plt.close()
    return pca.explained_variance_ratio_

import matplotlib.pyplot as plt
import seaborn as sns  # only for nicer default styles

# ─── New plotting helpers ───────────────────────────────────────────────────

def plot_linearity_curves(df_lin, lin_fits, outdir='linear_results'):
    """3×3 grid: each row=dye, each col=channel, showing data+fit overlay."""
    sns.set(style="whitegrid")
    dyes = ['red','yellow','green']
    channels = ['red','green','blue']
    fig, axes = plt.subplots(3, 3, figsize=(12,12), sharex=True, sharey=True)

    for i, dye in enumerate(dyes):
        sel = df_lin[df_lin[f'{dye}_vol']>0]
        xs = np.linspace(0,1,101).reshape(-1,1)
        for j, channel in enumerate(channels):
            ax = axes[i][j]
            meas = f'measured_{channel}'
            ax.scatter(sel[f'{dye}_frac'], sel[meas], alpha=0.6)
            slope = lin_fits.loc[(dye,channel),'slope']
            intercept = lin_fits.loc[(dye,channel),'intercept']
            ax.plot(xs, slope*xs + intercept, 'r-')
            ax.set_title(f'{dye.title()}→{channel.title()}')
            if i==2: ax.set_xlabel('fraction')
            if j==0: ax.set_ylabel('intensity')
    fig.tight_layout()
    fig.savefig(f'{outdir}/all_linear_fits.png')
    plt.close(fig)


def plot_mixture_nonlinearity(df_mix, lin_fits, df_all, outdir='mix_results'):
    Path(outdir).mkdir(exist_ok=True)
    # rebuild baseline once
    water = df_all[
        (df_all.red_vol==0)&(df_all.yellow_vol==0)&(df_all.green_vol==0)
    ]
    baseline = {ch: water[f'measured_{ch}'].mean()
                for ch in ['red','green','blue']}

    fig, axes = plt.subplots(1,3,figsize=(15,5))
    for ax, ch in zip(axes, ['red','green','blue']):
        y_true = df_mix[f'measured_{ch}']
        y_pred = baseline[ch] + sum(
            df_mix[f'{dye}_frac'] * lin_fits.loc[(dye,ch),'slope']
            for dye in ['red','yellow','green']
        )
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        ax.scatter(y_pred, y_true, alpha=0.6)
        m = max(y_true.max(), y_pred.max())
        ax.plot([0,m],[0,m],'k--',linewidth=1)
        ax.set_title(f'{ch.title()} RMSE={rmse:.1f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Measured')
    fig.tight_layout()
    fig.savefig(f'{outdir}/mixture_nonlinearity_intercept.png')
    plt.close(fig)


def plot_pca_scatter(df_all, pca, outdir='pca'):
    """
    2D PCA of all wells, but color each point by its actual measured RGB.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    sns.set(style="darkgrid")
    Path(outdir).mkdir(exist_ok=True)

    # 1) project into PC space
    rgb_cols = ['measured_red','measured_green','measured_blue']
    coords = pca.transform(df_all[rgb_cols])
    pc1, pc2 = coords[:,0], coords[:,1]

    # 2) normalize measured RGB to [0,1] for matplotlib
    rgb_norm = (df_all[rgb_cols].values / 255.0).clip(0,1)

    # 3) scatter with true colors
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(pc1, pc2, c=rgb_norm, alpha=0.8, edgecolor='k', linewidth=0.2)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA of all wells (colored by measured RGB)')

    # 4) plot PC2 loadings as arrows
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, ch in enumerate(['R','G','B']):
        dx, dy = loadings[i,0]*3, loadings[i,1]*3
        ax.arrow(0, 0, dx, dy, head_width=1.5, length_includes_head=True, color='k')
        ax.text(dx*1.1, dy*1.1, ch, color='k', fontsize=12, weight='bold')

    fig.tight_layout()
    fig.savefig(f'{outdir}/pca_scatter_truecolors.png', dpi=150)
    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────────

def main(csv_path):
    df = load_plate_data(csv_path)
    df = compute_fractions(df)
    df_lin, df_mix = split_experiments(df)

    print(f'Found {len(df_lin)} linear-space wells, {len(df_mix)} mixture wells.')
    lin_fits = analyze_linear(df_lin)
    print('Linear fits:')
    print(lin_fits)

    mix_stats = analyze_mixtures(df_mix, lin_fits)
    print('Mixture analysis:')
    print(mix_stats)

    mix_stats_water = analyze_mixtures_with_intercept(df_mix, lin_fits, df)
    print("Intercept analysis:\n", mix_stats_water)

    var_ratios = analyze_pca(df)
    print('PCA variance ratios (2 components):', var_ratios)

    plot_linearity_curves(df_lin, lin_fits)
    plot_mixture_nonlinearity(df_mix, lin_fits, df)
    # for PCA, re‐run PCA object from your analyze_pca or fit a fresh one:
    from sklearn.decomposition import PCA
    pca2 = PCA(n_components=2).fit(df[['measured_red','measured_green','measured_blue']])
    plot_pca_scatter(df, pca2)

    print('All plots saved to linear_results/, mix_results/, pca/')

if __name__ == '__main__':
    main("color_space_testing/ot_2_lighting/linspace_data.csv")
