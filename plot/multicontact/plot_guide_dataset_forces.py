"""
Plot contact forces from a guide_dataset_dyn_single.npz file for debugging.

Usage:
    python plot/multicontact/plot_guide_dataset_forces.py [path_to.npz]

Defaults to guide_dataset_dyn_single.npz in the working directory.
"""

import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NPZ_PATH = sys.argv[1] if len(sys.argv) > 1 else "guide_dataset_dyn_single.npz"

# Friction coefficient used in the planner (for cone check overlay)
MU = 0.7

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
d = np.load(NPZ_PATH, allow_pickle=True)

n_trials    = d['contact_forces_feet'].shape[0]
n_steps     = d['contact_forces_feet'].shape[1]
dt          = float(d['dt'])
t           = np.arange(n_steps) * dt

feet_names  = list(d['contact_forces_feet_names'])   # ['LF', 'RF']
hands_names = list(d['contact_forces_hands_names'])  # ['LH', 'RH']

# (n_trials, n_steps, 2, 6)  — feet 6D wrench
forces_feet  = d['contact_forces_feet']
# (n_trials, n_steps, 2, 3)  — hands 3D linear force
forces_hands = d['contact_forces_hands']
# (n_trials, n_steps, 4)  — [LF, RF, LH, RH]
contact_mask = d['contact_mask']

is_optimal = d['is_optimal']

print(f"Loaded {NPZ_PATH}")
print(f"  Trials : {n_trials}  |  steps : {n_steps}  |  dt : {dt:.4f} s  |  T : {t[-1]:.2f} s")
print(f"  Optimal: {is_optimal}")
print(f"  Feet force range  : [{forces_feet.min():.2f}, {forces_feet.max():.2f}] N/Nm")
print(f"  Hands force range : [{forces_hands.min():.2f}, {forces_hands.max():.2f}] N")


# ---------------------------------------------------------------------------
# Helper: shade contact-active regions on an axes
# ---------------------------------------------------------------------------
def shade_contact(ax, mask_1d, color, alpha=0.12, label=None):
    in_contact = False
    t_start = 0.
    for k in range(len(mask_1d)):
        active = mask_1d[k] > 0.5
        if active and not in_contact:
            t_start = t[k]
            in_contact = True
        elif not active and in_contact:
            ax.axvspan(t_start, t[k], color=color, alpha=alpha)
            in_contact = False
    if in_contact:
        ax.axvspan(t_start, t[-1], color=color, alpha=alpha, label=label)


# Colors per contact
COLORS = {
    'LF': 'tab:blue',
    'RF': 'tab:orange',
    'LH': 'tab:green',
    'RH': 'tab:red',
}

EE_MASK_IDX = {'LF': 0, 'RF': 1, 'LH': 2, 'RH': 3}


# ---------------------------------------------------------------------------
# Plot per trial
# ---------------------------------------------------------------------------
for trial in range(n_trials):
    tag = f"Trial {trial}  ({'OPTIMAL' if is_optimal[trial] else 'SUBOPTIMAL'})"

    # ---- Figure 1: Foot linear forces ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Foot Linear Forces (Fx, Fy, Fz)  —  {tag}", fontsize=12)

    ylabels = ['Fx [N]', 'Fy [N]', 'Fz [N]']
    for col, (ee, fi) in enumerate(zip(feet_names, range(2))):
        mask_idx = EE_MASK_IDX[ee]
        for row, comp in enumerate(range(3)):
            ax = axes[row, col]
            ax.plot(t, forces_feet[trial, :, fi, comp], color=COLORS[ee], linewidth=1.0)
            ax.axhline(0, color='k', linewidth=0.4, linestyle='--')
            # shade when this foot is active
            shade_contact(ax, contact_mask[trial, :, mask_idx], COLORS[ee], alpha=0.15)
            ax.set_ylabel(ylabels[row])
            if row == 0:
                ax.set_title(ee)
            if row == 2:
                ax.set_xlabel("time [s]")
            ax.grid(True, linewidth=0.3)

    plt.tight_layout()

    # ---- Figure 2: Foot moments ----
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    fig2.suptitle(f"Foot Moments (Mx, My, Mz)  —  {tag}", fontsize=12)

    ylabels_m = ['Mx [Nm]', 'My [Nm]', 'Mz [Nm]']
    for col, (ee, fi) in enumerate(zip(feet_names, range(2))):
        mask_idx = EE_MASK_IDX[ee]
        for row, comp in enumerate(range(3, 6)):
            ax = axes2[row, col]
            ax.plot(t, forces_feet[trial, :, fi, comp], color=COLORS[ee], linewidth=1.0)
            ax.axhline(0, color='k', linewidth=0.4, linestyle='--')
            shade_contact(ax, contact_mask[trial, :, mask_idx], COLORS[ee], alpha=0.15)
            ax.set_ylabel(ylabels_m[row])
            if row == 0:
                ax.set_title(ee)
            if row == 2:
                ax.set_xlabel("time [s]")
            ax.grid(True, linewidth=0.3)

    plt.tight_layout()

    # ---- Figure 3: Hand linear forces ----
    fig3, axes3 = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    fig3.suptitle(f"Hand Linear Forces (Fx, Fy, Fz)  —  {tag}", fontsize=12)

    for col, (ee, hi) in enumerate(zip(hands_names, range(2))):
        mask_idx = EE_MASK_IDX[ee]
        for row, comp in enumerate(range(3)):
            ax = axes3[row, col]
            ax.plot(t, forces_hands[trial, :, hi, comp], color=COLORS[ee], linewidth=1.0)
            ax.axhline(0, color='k', linewidth=0.4, linestyle='--')
            shade_contact(ax, contact_mask[trial, :, mask_idx], COLORS[ee], alpha=0.15)
            ax.set_ylabel(ylabels[row])
            if row == 0:
                ax.set_title(ee)
            if row == 2:
                ax.set_xlabel("time [s]")
            ax.grid(True, linewidth=0.3)

    plt.tight_layout()

    # ---- Figure 4: Normal force + friction check ----
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    fig4.suptitle(f"Normal Force (Fz) and Friction Ratio |Ft|/Fz  —  {tag}", fontsize=12)

    # Feet: normal = Fz (index 2), tangential = sqrt(Fx²+Fy²)
    for col, (ee, fi) in enumerate(zip(feet_names, range(2))):
        mask_idx = EE_MASK_IDX[ee]
        Fz  = forces_feet[trial, :, fi, 2]
        Ft  = np.sqrt(forces_feet[trial, :, fi, 0]**2 + forces_feet[trial, :, fi, 1]**2)
        # friction ratio (avoid div by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(np.abs(Fz) > 1.0, Ft / np.abs(Fz), np.nan)

        ax_n = axes4[0, col]
        ax_n.plot(t, Fz, color=COLORS[ee], linewidth=1.0, label='Fz')
        ax_n.axhline(0, color='k', linewidth=0.4, linestyle='--')
        shade_contact(ax_n, contact_mask[trial, :, mask_idx], COLORS[ee], alpha=0.15)
        ax_n.set_title(ee)
        ax_n.set_ylabel('Fz [N]')
        ax_n.grid(True, linewidth=0.3)

        ax_f = axes4[1, col]
        ax_f.plot(t, ratio, color=COLORS[ee], linewidth=1.0, label='|Ft|/Fz')
        ax_f.axhline(MU, color='crimson', linewidth=0.8, linestyle='--', label=f'μ={MU}')
        shade_contact(ax_f, contact_mask[trial, :, mask_idx], COLORS[ee], alpha=0.15)
        ax_f.set_ylabel('|Ft|/Fz')
        ax_f.set_xlabel('time [s]')
        ax_f.legend(fontsize=8)
        ax_f.grid(True, linewidth=0.3)
        ax_f.set_ylim(0, 2.0)

    plt.tight_layout()

    # ---- Figure 5: All forces on one timeline ----
    fig5, axes5 = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig5.suptitle(f"All Contact Forces (Fz feet, Fx hands normal)  —  {tag}", fontsize=12)

    # Row 0: LF Fz
    axes5[0].plot(t, forces_feet[trial, :, 0, 2], color=COLORS['LF'], linewidth=1.0)
    shade_contact(axes5[0], contact_mask[trial, :, 0], COLORS['LF'], alpha=0.15)
    axes5[0].set_ylabel('LF Fz [N]')
    axes5[0].axhline(0, color='k', linewidth=0.4, linestyle='--')
    axes5[0].grid(True, linewidth=0.3)

    # Row 1: RF Fz
    axes5[1].plot(t, forces_feet[trial, :, 1, 2], color=COLORS['RF'], linewidth=1.0)
    shade_contact(axes5[1], contact_mask[trial, :, 1], COLORS['RF'], alpha=0.15)
    axes5[1].set_ylabel('RF Fz [N]')
    axes5[1].axhline(0, color='k', linewidth=0.4, linestyle='--')
    axes5[1].grid(True, linewidth=0.3)

    # Row 2: LH forces
    axes5[2].plot(t, forces_hands[trial, :, 0, 0], color=COLORS['LH'], linewidth=1.0, label='Fx')
    axes5[2].plot(t, forces_hands[trial, :, 0, 1], color=COLORS['LH'], linewidth=1.0, linestyle='--', label='Fy', alpha=0.6)
    axes5[2].plot(t, forces_hands[trial, :, 0, 2], color=COLORS['LH'], linewidth=1.0, linestyle=':', label='Fz', alpha=0.6)
    shade_contact(axes5[2], contact_mask[trial, :, 2], COLORS['LH'], alpha=0.15)
    axes5[2].set_ylabel('LH Force [N]')
    axes5[2].axhline(0, color='k', linewidth=0.4, linestyle='--')
    axes5[2].legend(fontsize=7, loc='upper right')
    axes5[2].grid(True, linewidth=0.3)

    # Row 3: RH forces
    axes5[3].plot(t, forces_hands[trial, :, 1, 0], color=COLORS['RH'], linewidth=1.0, label='Fx')
    axes5[3].plot(t, forces_hands[trial, :, 1, 1], color=COLORS['RH'], linewidth=1.0, linestyle='--', label='Fy', alpha=0.6)
    axes5[3].plot(t, forces_hands[trial, :, 1, 2], color=COLORS['RH'], linewidth=1.0, linestyle=':', label='Fz', alpha=0.6)
    shade_contact(axes5[3], contact_mask[trial, :, 3], COLORS['RH'], alpha=0.15)
    axes5[3].set_ylabel('RH Force [N]')
    axes5[3].set_xlabel('time [s]')
    axes5[3].axhline(0, color='k', linewidth=0.4, linestyle='--')
    axes5[3].legend(fontsize=7, loc='upper right')
    axes5[3].grid(True, linewidth=0.3)

    plt.tight_layout()

    # ---- Figure 6: Contact schedule (contact_mask) ----
    EE_SCHEDULE_ORDER = ['LF', 'RF', 'LH', 'RH']
    fig6, ax6 = plt.subplots(figsize=(14, 3))
    fig6.suptitle(f"Contact Schedule  —  {tag}", fontsize=12)

    y_ticks, y_labels = [], []
    for row, ee in enumerate(EE_SCHEDULE_ORDER):
        mask_idx = EE_MASK_IDX[ee]
        mask_1d  = contact_mask[trial, :, mask_idx]
        y_base   = row
        # filled bar at y_base when active
        ax6.fill_between(t, y_base, y_base + 0.8, where=mask_1d > 0.5,
                         color=COLORS[ee], alpha=0.85, linewidth=0)
        # raw mask value as a thin line so partial/intermediate values are visible
        ax6.plot(t, y_base + 0.8 * mask_1d, color=COLORS[ee],
                 linewidth=0.6, alpha=0.4)
        y_ticks.append(y_base + 0.4)
        y_labels.append(ee)

    ax6.set_yticks(y_ticks)
    ax6.set_yticklabels(y_labels, fontsize=10)
    ax6.set_xlim(t[0], t[-1])
    ax6.set_ylim(-0.1, len(EE_SCHEDULE_ORDER))
    ax6.set_xlabel("time [s]")
    ax6.grid(axis='x', linewidth=0.3)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['left'].set_visible(False)
    ax6.tick_params(left=False)

    plt.tight_layout()

plt.show()
