# plot_utils.py
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lake_mdp import UP, RIGHT, DOWN, LEFT

# Import env action constants (works if they are defined; otherwise we compare by string)
try:
    from lake_mdp import UP, RIGHT, DOWN, LEFT, ABSORB
except Exception:
    UP, RIGHT, DOWN, LEFT, ABSORB = "UP", "RIGHT", "DOWN", "LEFT", "ABSORB"

# Map action names -> arrow glyphs (ASCII-friendly fallbacks provided)
ACTION_TO_ARROW = {
    "UP": "↑",
    "RIGHT": "→",
    "DOWN": "↓",
    "LEFT": "←",
}


def _action_name(a):
    """Normalize an action object to 'UP'/'RIGHT'/'DOWN'/'LEFT' when possible."""
    if a == UP:
        return "UP"
    if a == RIGHT:
        return "RIGHT"
    if a == DOWN:
        return "DOWN"
    if a == LEFT:
        return "LEFT"
    # Fallback for string-like actions
    if isinstance(a, str):
        s = a.strip().upper()
        if s in ACTION_TO_ARROW:
            return s
    return None  # unknown / ABSORB / other


def plot_policy(policy, ax=None):
    """
    Plot the LakeMDP board using only the policy:
      - Extract mdp via policy.mdp
      - Color S (orange), H (blue), G (green); all other cells white
      - For every non-(S/H/G) cell, draw the arrow dictated by policy(s)

    Assumptions:
      - policy is callable on a *state* s and exposes `policy.mdp`
      - `policy.mdp.grid` is a list[list[str]] with characters in {'S','F','H','G'}
      - states are addressed as (i, j) tuples when calling policy((i, j))
    """
    mdp = policy.mdp
    grid = mdp.grid
    m = len(grid)
    n = len(grid[0]) if m else 0

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
        ax.set_facecolor("white")

    # Iterate rows/cols once: draw cell + add character (letter or arrow)
    for i in range(m):
        for j in range(n):
            ch = grid[i][j]

            # choose cell color
            if ch == "S":
                color, txt, txt_color = (1.0, 0.6, 0.0), "S", "white"  # orange
            elif ch == "H":
                color, txt, txt_color = (0.2, 0.4, 1.0), "H", "white"  # blue
            elif ch == "G":
                color, txt, txt_color = (0.0, 0.7, 0.0), "G", "white"  # green
            else:
                # Free cell: white, text decided by the policy
                color, txt, txt_color = (1.0, 1.0, 1.0), None, "black"

            # draw the box
            ax.add_patch(
                plt.Rectangle(
                    (j, i), 1, 1, facecolor=color, edgecolor="lightgray", linewidth=1.0
                )
            )

            # decide the character to place
            if txt is None:
                # ask the policy for this state's action, then map to arrow
                s = (i, j)
                try:
                    a = policy((s, ch))
                    print(f"Policy action at state {s}: {a}")
                except Exception:
                    print(f"Policy action at state {s}: <error>")
                    a = None
                name = _action_name(a)
                txt = ACTION_TO_ARROW.get(name, "·")  # dot if unknown / ABSORB

            # put the character in the center
            ax.text(
                j + 0.5,
                i + 0.55,
                txt,
                ha="center",
                va="center",
                fontsize=14,
                color=txt_color,
                weight="bold",
            )

    # tidy axes
    ax.set_xlim(0, n)
    ax.set_ylim(m, 0)  # flip to match grid layout
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    return ax


# ---------- Advantage plotting utilities ----------
def plot_advantages_gif(
    mdp,
    advantages,
    out_path="advantages.gif",
    fps=2,
    use_global_scale=True,
    arrow_scale=3.0,
    axes_margin=0.12,
):
    m = len(mdp.grid)
    n = len(mdp.grid[0]) if m else 0

    action_index = {UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3}

    # Build frames: (m, n, 4) per time step
    frames = []
    for adv in advantages:
        vals = np.zeros((m, n, 4), dtype=float)
        for ((pos, sym), a), ad in adv.items():
            if sym == "⊥":
                continue
            i, j = pos
            if a in action_index:
                vals[i, j, action_index[a]] = float(ad)
        frames.append(vals)
    if not frames:
        raise ValueError("No frames found in 'advantages'.")

    # Color scale
    if use_global_scale:
        data_min = np.min([f.min() for f in frames])
        data_max = np.max([f.max() for f in frames])
    else:
        data_min = frames[0].min()
        data_max = frames[0].max()

    cmap = plt.cm.RdYlGn

    fig, axes = plt.subplots(m, n, figsize=(max(7, n * 2.7), max(7, m * 2.7)))
    axes = np.atleast_2d(axes)
    plt.subplots_adjust(wspace=0.6, hspace=0.6)

    arrow_patches = [[None for _ in range(n)] for _ in range(m)]
    norms = []

    for i in range(m):
        for j in range(n):
            ax = axes[i, j]
            ch = mdp.grid[i][j]

            # S/H/G cells: colored box + letter (no arrows)
            if ch in {"S", "H", "G"}:
                if ch == "S":
                    color, txt, txt_color = (1.0, 0.6, 0.0), "S", "white"
                elif ch == "H":
                    color, txt, txt_color = (0.2, 0.4, 1.0), "H", "white"
                else:
                    color, txt, txt_color = (0.0, 0.7, 0.0), "G", "white"

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect("equal", adjustable="box")
                ax.axis("off")

                ax.add_patch(
                    plt.Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="lightgray",
                        linewidth=1.0,
                    )
                )
                ax.text(
                    0.5,
                    0.5,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=14,
                    color=txt_color,
                    weight="bold",
                )
                arrow_patches[i][j] = None
                continue

            # Free cells: axes limits with margin so heads don't clip
            ax.set_xlim(-1 - axes_margin, 1 + axes_margin)
            ax.set_ylim(-1 - axes_margin, 1 + axes_margin)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

            # Crosshair
            ax.plot(
                [-0.8, 0.8],
                [0, 0],
                linewidth=1,
                alpha=0.6,
                solid_capstyle="round",
                color="black",
            )
            ax.plot(
                [0, 0],
                [-0.8, 0.8],
                linewidth=1,
                alpha=0.6,
                solid_capstyle="round",
                color="black",
            )

            up, right, down, left = frames[0][i, j]

            def _frac(v, vmin=data_min, vmax=data_max):
                return 0.5 if vmax == vmin else (v - vmin) / (vmax - vmin)

            # Bigger heads + avoid clipping
            hw = 0.15 * arrow_scale
            hl = 0.15 * arrow_scale
            arrow_kwargs = dict(
                head_width=hw,
                head_length=hl,
                ec="black",
                clip_on=False,  # <-- key fix: don't clip to axes
                length_includes_head=False,  # <-- keep arrow inside the limits
            )

            patches = {}
            patches["up"] = ax.arrow(0, 0, 0, 0.6, fc=cmap(_frac(up)), **arrow_kwargs)
            patches["right"] = ax.arrow(
                0, 0, 0.6, 0, fc=cmap(_frac(right)), **arrow_kwargs
            )
            patches["down"] = ax.arrow(
                0, 0, 0, -0.6, fc=cmap(_frac(down)), **arrow_kwargs
            )
            patches["left"] = ax.arrow(
                0, 0, -0.6, 0, fc=cmap(_frac(left)), **arrow_kwargs
            )
            arrow_patches[i][j] = patches

            # Per-subplot colorbar (free cells only)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.15)
            norm = matplotlib.colors.Normalize(vmin=data_min, vmax=data_max)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = plt.colorbar(sm, cax=cax)
            cb.set_ticks([data_min, (data_min + data_max) / 2.0, data_max])
            cb.ax.tick_params(labelsize=8)
            norms.append((sm, cb))

    def recolor(patch, value, vmin, vmax):
        if vmax == vmin:
            patch.set_facecolor(cmap(0.5))
        else:
            patch.set_facecolor(cmap((value - vmin) / (vmax - vmin)))

    def update(t):
        if use_global_scale:
            vmin, vmax = data_min, data_max
        else:
            vmin, vmax = frames[t].min(), frames[t].max()
            for sm, cb in norms:
                sm.set_clim(vmin=vmin, vmax=vmax)
                cb.set_ticks([vmin, (vmin + vmax) / 2.0, vmax])
                cb.update_normal(sm)

        for i in range(m):
            for j in range(n):
                patches = arrow_patches[i][j]
                if not patches:
                    continue
                up, right, down, left = frames[t][i, j]
                recolor(patches["up"], up, vmin, vmax)
                recolor(patches["right"], right, vmin, vmax)
                recolor(patches["down"], down, vmin, vmax)
                recolor(patches["left"], left, vmin, vmax)

        fig.suptitle(f"Advantages – Frame {t+1}/{len(frames)}", fontsize=12)
        return []

    anim = FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // max(fps, 1), blit=False
    )
    writer = PillowWriter(fps=fps, metadata={"loop": 0})
    anim.save(out_path, writer=writer)
    plt.close(fig)
    return out_path
