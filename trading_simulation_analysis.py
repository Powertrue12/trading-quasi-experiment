import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ----------------------------------------------------------------------
# 1. Data generation
# ----------------------------------------------------------------------
def generate_dataset(seed: int = 12345,
                     n_capital: int = 73,
                     n_derivatives: int = 84,
                     n_weeks: int = 10) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    group_labels = np.array(["Capital"] * n_capital + ["Derivatives"] * n_derivatives)

    weekly_returns = []
    leverage_used = []
    stop_loss = []
    trade_count = []
    risk_literacy = []
    confidence = []

    for group in group_labels:
        if group == "Capital":
            mean_ret = 0.002
            sd_ret = 0.01
            leverage_mean = 1.5
            leverage_sd = 0.5
            stop_prob = 0.7
            trades_mean = 30
            trades_sd = 10
            literacy_mean = 5.5
            literacy_sd = 1.0
            confidence_mean = 5.0
            confidence_sd = 1.5
        else:  # Derivatives
            mean_ret = 0.004
            sd_ret = 0.02
            leverage_mean = 3.5
            leverage_sd = 1.0
            stop_prob = 0.3
            trades_mean = 60
            trades_sd = 15
            literacy_mean = 5.0
            literacy_sd = 1.2
            confidence_mean = 6.0
            confidence_sd = 1.5

        # Weekly returns
        wr = rng.normal(loc=mean_ret, scale=sd_ret, size=n_weeks)

        # Inject occasional “crash” in derivatives to mimic blow-ups
        if group == "Derivatives" and rng.random() < 0.08:
            crash_week = rng.integers(0, n_weeks)
            wr[crash_week] = rng.normal(loc=-0.4, scale=0.05)

        weekly_returns.append(wr)
        leverage_used.append(max(0.5, rng.normal(leverage_mean, leverage_sd)))
        stop_loss.append(rng.random() < stop_prob)
        trade_count.append(int(max(1, rng.normal(trades_mean, trades_sd))))
        risk_literacy.append(float(np.clip(rng.normal(literacy_mean, literacy_sd), 0, 10)))
        confidence.append(float(np.clip(rng.normal(confidence_mean, confidence_sd), 0, 10)))

    total_return = np.array([wr.sum() for wr in weekly_returns])

    # Maximum drawdown
    max_drawdowns = []
    for wr in weekly_returns:
        cum_returns = np.cumsum(wr)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns - running_max
        max_drawdowns.append(drawdowns.min())
    max_drawdowns = np.array(max_drawdowns)

    # Behavioural risk profiles
    risk_profiles = []
    for lev, draw, stop in zip(leverage_used, max_drawdowns, stop_loss):
        if lev > 4 or draw < -0.3 or (not stop):
            risk_profiles.append("Aggressive")
        elif lev > 2.5 or draw < -0.15:
            risk_profiles.append("Moderate")
        else:
            risk_profiles.append("Conservative")

    blow_up = total_return < -0.2

    df = pd.DataFrame({
        "group": group_labels,
        "total_return": total_return,
        "max_drawdown": max_drawdowns,
        "leverage_used": leverage_used,
        "stop_loss": stop_loss,
        "trade_count": trade_count,
        "risk_literacy": risk_literacy,
        "confidence": confidence,
        "risk_profile": risk_profiles,
        "blow_up": blow_up,
    })

    # Store individual weekly returns as separate columns
    for i in range(n_weeks):
        df[f"week_{i+1}"] = [wr[i] for wr in weekly_returns]

    return df


# ----------------------------------------------------------------------
# 2. Descriptive statistics and group comparisons
# ----------------------------------------------------------------------
def descriptive_stats(df: pd.DataFrame, by: str = "group") -> pd.DataFrame:
    """Means, SD and 95% t-intervals for numeric variables by group."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []

    for var in numeric_cols:
        for grp, sub in df.groupby(by):
            values = sub[var].dropna()
            n = values.size
            mean = values.mean()
            std = values.std(ddof=1)
            se = std / np.sqrt(n) if n > 0 else np.nan
            if n > 1:
                ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
            else:
                ci_low, ci_high = np.nan, np.nan

            rows.append({
                "variable": var,
                "group": grp,
                "mean": mean,
                "std": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": n,
            })

    return pd.DataFrame(rows)


def group_comparison_tests(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Welch t-tests and Hedges’ g for selected variables."""
    results = []
    g1, g2 = "Capital", "Derivatives"   # fixed order for reproducibility

    for var in variables:
        x = df.loc[df["group"] == g1, var].dropna()
        y = df.loc[df["group"] == g2, var].dropna()

        t_stat, p_val = stats.ttest_ind(x, y, equal_var=False)
        s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
        n1, n2 = len(x), len(y)

        pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        d = (x.mean() - y.mean()) / pooled_sd if pooled_sd > 0 else np.nan
        correction = 1 - (3 / (4 * (n1 + n2) - 9))
        hedges_g = d * correction

        df_num = (s1 / n1 + s2 / n2) ** 2 / (
            (s1 ** 2) / ((n1 ** 2) * (n1 - 1)) +
            (s2 ** 2) / ((n2 ** 2) * (n2 - 1))
        )

        results.append({
            "variable": var,
            "t_stat": t_stat,
            "df": df_num,
            "p_value": p_val,
            "hedges_g": hedges_g,
        })

    return pd.DataFrame(results)


# ----------------------------------------------------------------------
# 3. Regression models
# ----------------------------------------------------------------------
def run_regressions(df: pd.DataFrame) -> dict:
    """Fit OLS (total return) and Logit (blow-up) models."""
    df = df.copy()
    df["group_code"] = (df["group"] == "Derivatives").astype(int)
    df["stop_loss_num"] = df["stop_loss"].astype(int)
    df["blow_up_numeric"] = df["blow_up"].astype(int)

    ols_model = smf.ols(
        "total_return ~ group_code + risk_literacy + confidence + leverage_used + stop_loss_num",
        data=df,
    ).fit()

    # Logit with robust solver and more iterations (reduces convergence warnings)
    logit_model = sm.Logit(
        df["blow_up_numeric"],
        sm.add_constant(df[["group_code", "risk_literacy", "confidence", "leverage_used", "stop_loss_num"]]),
    ).fit(method="lbfgs", maxiter=1000, disp=False)

    return {"ols": ols_model, "logit": logit_model}


# ----------------------------------------------------------------------
# 4. Figures
# ----------------------------------------------------------------------
def plot_distributions(df: pd.DataFrame, output_dir: str = ".") -> list:
    """Generate and save Figures 1–3."""
    sns.set(style="whitegrid")
    paths = []

    # Figure 1 – total returns
    plt.figure(figsize=(6, 4))
    sns.histplot(
        data=df,
        x="total_return",
        hue="group",
        element="step",
        stat="density",
        common_norm=False,
        bins=30,
        alpha=0.4,
    )
    sns.kdeplot(data=df, x="total_return", hue="group", common_norm=False)
    plt.xlabel("Total return")
    plt.title("Distribution of Total Returns by Course")
    plt.tight_layout()
    path1 = f"{output_dir}/fig_distribution_total_return.png"
    plt.savefig(path1, dpi=300)
    plt.close()
    paths.append(path1)

    # Figure 2 – leverage by course
    plt.figure(figsize=(6, 4))
    ax = sns.boxplot(data=df, x="group", y="leverage_used")
    sns.stripplot(data=df, x="group", y="leverage_used",
                  color="black", size=2, alpha=0.3, ax=ax)
    plt.ylabel("Leverage used")
    plt.title("Leverage Usage by Course")
    plt.tight_layout()
    path2 = f"{output_dir}/fig_box_leverage.png"
    plt.savefig(path2, dpi=300)
    plt.close()
    paths.append(path2)

    # Figure 3 – risk profile distribution with Wilson 95% CIs
    profile_counts = df.groupby(["group", "risk_profile"]).size().reset_index(name="count")
    total_counts = df.groupby("group").size().reset_index(name="total")
    profile_counts = profile_counts.merge(total_counts, on="group")
    profile_counts["proportion"] = profile_counts["count"] / profile_counts["total"]

    def wilson_ci(k, n, alpha=0.05):
        z = stats.norm.ppf(1 - alpha / 2)
        phat = k / n
        denom = 1 + z ** 2 / n
        center = (phat + z ** 2 / (2 * n)) / denom
        half_width = z * np.sqrt((phat * (1 - phat) + z ** 2 / (4 * n)) / n) / denom
        return center - half_width, center + half_width

    cis = [wilson_ci(k, n) for k, n in zip(profile_counts["count"], profile_counts["total"])]
    profile_counts["ci_low"] = [c[0] for c in cis]
    profile_counts["ci_high"] = [c[1] for c in cis]

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=profile_counts,
        x="risk_profile",
        y="proportion",
        hue="group",
        ci=None,
    )
    # Error bars centred on each bar
    for bar, (_, row) in zip(ax.patches, profile_counts.iterrows()):
        x = bar.get_x() + bar.get_width() / 2
        ax.plot([x, x], [row["ci_low"], row["ci_high"]], color="black")

    ax.set_ylabel("Proportion")
    ax.set_title("Risk Profile Distribution with 95% CIs")
    plt.tight_layout()
    path3 = f"{output_dir}/fig_risk_profile_ci.png"
    plt.savefig(path3, dpi=300)
    plt.close()
    paths.append(path3)

    return paths


def margins_plot(model: sm.Logit, df: pd.DataFrame, output_path: str) -> None:
    """Figure 4 – marginal effect of course type on blow-up probability."""
    means = df[["risk_literacy", "confidence", "leverage_used", "stop_loss"]].copy()
    means["stop_loss"] = means["stop_loss"].astype(int)
    means = means.mean()

    X = pd.DataFrame({
        "const": 1.0,
        "group_code": [0, 1],
        "risk_literacy": [means["risk_literacy"]] * 2,
        "confidence": [means["confidence"]] * 2,
        "leverage_used": [means["leverage_used"]] * 2,
        "stop_loss_num": [means["stop_loss"]] * 2,
    })

    preds = model.predict(X)
    cov = model.cov_params()

    se_preds = []
    for i, row in X.iterrows():
        p = preds.iloc[i]
        grad = p * (1 - p) * row.values  # derivative of logit link
        var = grad @ cov.values @ grad.T
        se_preds.append(np.sqrt(var))

    z = stats.norm.ppf(0.975)
    lower = preds - z * np.array(se_preds)
    upper = preds + z * np.array(se_preds)

    plt.figure(figsize=(6, 4))
    labels = ["Capital", "Derivatives"]
    plt.errorbar(labels, preds, yerr=[preds - lower, upper - preds],
                 fmt="o", capsize=5)
    plt.ylabel("Predicted probability of blow-up")
    plt.title("Marginal Effect of Course on Blow-Up Probability")
    plt.ylim(0, float(max(upper)) + 0.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ----------------------------------------------------------------------
# 5. Main entry point
# ----------------------------------------------------------------------
def main():
    seed = 12345
    print(f"Random seed used: {seed}")

    df = generate_dataset(seed)
    df.to_csv("trading_simulation_data.csv", index=False)

    desc = descriptive_stats(df)
    desc.to_csv("table_descriptives.csv", index=False)

    vars_to_compare = ["total_return", "max_drawdown", "leverage_used", "trade_count"]
    comparisons = group_comparison_tests(df, vars_to_compare)
    comparisons.to_csv("table_group_comparisons.csv", index=False)

    models = run_regressions(df)
    with open("ols_summary.txt", "w") as fh:
        fh.write(models["ols"].summary().as_text())
    with open("logit_summary.txt", "w") as fh:
        fh.write(models["logit"].summary().as_text())

    plot_distributions(df)
    margins_plot(models["logit"], df, "fig_margins.png")

    print("Synthetic data generation and analysis complete.")


if __name__ == "__main__":
    main()
