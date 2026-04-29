import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Plot: 2 Objs and pareto front
def plot_obj_2d(F, xlim=(0, 1), ylim=(0, 1)):
    n_obj = F.shape[1]
    if n_obj == 2:
        nds = NonDominatedSorting()
        front_idx = nds.do(F, only_non_dominated_front=True)

        pareto_F = F[front_idx]
        non_pareto_F = np.delete(F, front_idx, axis=0)

        fig = go.Figure(
            data=go.Scatter(
                x=F[:, 0],
                y=F[:, 1],
                mode='markers',
                name='Objective Values',
                marker=dict(size=6, color='#FF7F0E', opacity=0.7)
            )
        )
        fig.add_trace(go.Scatter(
            x=pareto_F[:, 0],
            y=pareto_F[:, 1],
            mode='markers',
            name='Pareto Front',
            marker=dict(size=7, color='#B07AA1', opacity=0.9, symbol='diamond')
        ))
        fig.update_layout(
            xaxis_title='f1',
            yaxis_title='f2',
            width=600,
            height=600,
            xaxis=dict(range=list(xlim)),
            yaxis=dict(range=list(ylim))
        )
        fig.show()


def plot_z_score(y_test, pred_mean, pred_std, bins=30, eps=1e-12):
    y_test = np.asarray(y_test)
    pred_mean = np.asarray(pred_mean)
    pred_std = np.asarray(pred_std)

    z = (y_test - pred_mean) / np.maximum(pred_std, eps)

    n_obj = z.shape[1]
    fig, axes = plt.subplots(1, n_obj, figsize=(5 * n_obj, 4), sharey=True)

    if n_obj == 1:
        axes = [axes]

    for i in range(n_obj):
        axes[i].hist(z[:, i], bins=bins)
        axes[i].set_title(f"Z distribution - f{i+1}")
        axes[i].set_xlabel("z")

    axes[0].set_ylabel("count")
    plt.tight_layout()
    plt.show()


def plot_y_true_pred(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true[:, 0], y_true[:, 1], label="y_true")
    plt.scatter(y_pred[:, 0], y_pred[:, 1], label="y_pred")
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_hv_history(
    results,
    title="HV over Generations",
    figsize=(7, 6),
    line_width=2.0,
    marker_size=4,
    tick_fontsize=12,
    label_fontsize=12,
    title_fontsize=14,
    legend_fontsize=11,
    show_plot=True,
    save_svg=True,
    svg_path="hv_curve.svg",
    show_legend=True,
    show_axis_labels=True,
    x_label="Generation",
    y_label="HV",
    xlim=(1, 100),
    ylim=(0, 1.3)
):
    gen_list = np.asarray(results["gen_history"])
    hv_sur_list = np.asarray(results["hv_sur_history"], dtype=float)
    hv_real_list = np.asarray(results["hv_real_history"], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    sur_label = "HV surrogate" if show_legend else None
    real_label = "HV real" if show_legend else None

    ax.plot(
        gen_list,
        hv_sur_list,
        marker='s',
        color="#1565C0",
        linewidth=line_width,
        markersize=marker_size,
        label=sur_label
    )

    ax.plot(
        gen_list,
        hv_real_list,
        marker='s',
        color="#D55E00",
        linewidth=line_width,
        markersize=marker_size,
        label=real_label
    )

    if show_axis_labels:
        ax.set_xlabel(x_label, fontsize=label_fontsize)
        ax.set_ylabel(y_label, fontsize=label_fontsize)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(axis='both', labelsize=tick_fontsize)

    if show_legend:
        ax.legend(fontsize=legend_fontsize)

    plt.tight_layout()

    if save_svg:
        plt.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"Figure saved as SVG: {svg_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
