import uproot
import matplotlib.pyplot as plt
import numpy as np

def muon_range_vs_energy(filename):
    with uproot.open(filename) as file:
        tree = file["output"]
        track_x   = tree["trackPosX"].array()
        track_y   = tree["trackPosY"].array()
        track_z   = tree["trackPosZ"].array()
        track_pdg = tree["trackPDG"].array()
        track_ke  = tree["trackKE"].array()

    energies = []
    ranges   = []

    for event_index in range(len(track_x)):
        ev_x   = track_x[event_index]
        ev_y   = track_y[event_index]
        ev_z   = track_z[event_index]
        ev_pdg = track_pdg[event_index]
        ev_ke  = track_ke[event_index]

        for track_idx in range(len(ev_x)):
            pdg = int(ev_pdg[track_idx])
            if pdg not in (13, -13):
                continue

            x_steps = np.array(ev_x[track_idx])
            y_steps = np.array(ev_y[track_idx])
            z_steps = np.array(ev_z[track_idx])
            ke_steps = np.array(ev_ke[track_idx])

            if len(x_steps) < 2:
                continue

            # Initial KE at first step
            initial_ke = ke_steps[0]

            # Total path length by summing step distances
            dx = np.diff(x_steps)
            dy = np.diff(y_steps)
            dz = np.diff(z_steps)
            path_length = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))

            energies.append(initial_ke)
            ranges.append(path_length)
            break  # only take the first muon track per event

    energies = np.array(energies)
    ranges   = np.array(ranges)

    # Sort by energy for clean plotting
    sort_idx = np.argsort(energies)
    energies = energies[sort_idx]
    ranges   = ranges[sort_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(energies, ranges, alpha=0.5, s=10, label='Individual muons')

    # Bin the data to get mean range per energy bin
    bins = np.linspace(energies.min(), energies.max(), 20)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means   = []
    bin_stds    = []
    for i in range(len(bins) - 1):
        mask = (energies >= bins[i]) & (energies < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(ranges[mask].mean())
            bin_stds.append(ranges[mask].std())
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    bin_means = np.array(bin_means)
    bin_stds  = np.array(bin_stds)

    ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
                fmt='o-', color='red', linewidth=2,
                capsize=4, label='Mean ± std per bin')

    ax.set_xlabel('Initial Muon KE (MeV)')
    ax.set_ylabel('Muon Path Length (mm)')
    ax.set_title('Muon Range vs Initial Kinetic Energy')
    ax.axhline(y=160, color='blue',  linestyle='--', linewidth=1.5, label='Scintillator exit point (160mm)')
    ax.axhline(y=180, color='gray',  linestyle='--', linewidth=1.5, label='Rock Vessel exit point (180mm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return energies, ranges

if __name__ == "__main__":
    energies, ranges = muon_range_vs_energy("../SimData/muon2_combined.root")