import uproot
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# PDG code to color and label mapping
PDG_COLORS = {
    13:   ('blue',    'mu-'),
    -13:  ('cyan',    'mu+'),
    11:   ('red',     'e-'),
    -11:  ('orange',  'e+'),
    22:   ('green',   'gamma'),
    2212: ('purple',  'proton'),
    2112: ('brown',   'neutron'),
    211:  ('pink',    'pi+'),
    -211: ('magenta', 'pi-'),
    12:   ('yellow',  'nu_e'),
    -12:  ('olive',   'anti-nu_e'),
    14:   ('lime',    'nu_mu'),
    -14:  ('teal',    'anti-nu_mu'),
}

def draw_box(ax, size, color, label, alpha=0.1, linewidth=1.5):
    """Draw a wireframe box centered at origin with given half-size."""
    s = size / 2
    # 8 corners of the box
    corners = np.array([
        [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s]
    ])
    # 6 faces
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # bottom
        [corners[4], corners[5], corners[6], corners[7]],  # top
        [corners[0], corners[1], corners[5], corners[4]],  # front
        [corners[2], corners[3], corners[7], corners[6]],  # back
        [corners[0], corners[3], corners[7], corners[4]],  # left
        [corners[1], corners[2], corners[6], corners[5]],  # right
    ]
    poly = Poly3DCollection(faces, alpha=alpha, linewidths=linewidth,
                             edgecolors=color, facecolors=color, label=label)
    ax.add_collection3d(poly)

def plot_event(ax, track_x, track_y, track_z, track_pdg, track_ke, event_index):
    ax.cla()

    # Draw detector geometry
    draw_box(ax, size=180, color='gray',  label='Rock Vessel',   alpha=0.05)
    draw_box(ax, size=160, color='blue',  label='Scintillator',  alpha=0.05)

    ev_x   = track_x[event_index]
    ev_y   = track_y[event_index]
    ev_z   = track_z[event_index]
    ev_pdg = track_pdg[event_index]
    ev_ke  = track_ke[event_index]

    num_tracks = len(ev_x)
    plotted_labels = set()
    plotted_labels.add('Rock Vessel')
    plotted_labels.add('Scintillator')

    # Find the muon's initial KE for the title
    muon_ke = None
    for track_idx in range(num_tracks):
        pdg = int(ev_pdg[track_idx])
        if pdg in (13, -13):
            ke_steps = np.array(ev_ke[track_idx])
            if len(ke_steps) > 0:
                muon_ke = ke_steps[0]
            break

    for track_idx in range(num_tracks):
        x_steps = np.array(ev_x[track_idx])
        y_steps = np.array(ev_y[track_idx])
        z_steps = np.array(ev_z[track_idx])
        pdg     = int(ev_pdg[track_idx])

        color, label = PDG_COLORS.get(pdg, ('gray', f'PDG:{pdg}'))

        if label not in plotted_labels:
            ax.plot(x_steps, y_steps, z_steps, marker='o', markersize=2,
                    linestyle='-', alpha=0.7, color=color, label=label)
            plotted_labels.add(label)
        else:
            ax.plot(x_steps, y_steps, z_steps, marker='o', markersize=2,
                    linestyle='-', alpha=0.7, color=color)

        if len(x_steps) > 0:
            ax.scatter(x_steps[0], y_steps[0], z_steps[0],
                       color='black', s=10, zorder=5)

    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.set_zlabel('Z Position (mm)')

    # Set axis limits to world volume
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_zlim(-200, 200)

    if muon_ke is not None:
        ax.set_title(f'3D Particle Tracks - Event {event_index} / {len(track_x)-1}'
                     f'\nMuon Initial KE: {muon_ke:.1f} MeV  |  ← → to navigate')
    else:
        ax.set_title(f'3D Particle Tracks - Event {event_index} / {len(track_x)-1}'
                     f'\nNo muon found  |  ← → to navigate')

    ax.legend()
    plt.draw()

def browse_events(filename):
    with uproot.open(filename) as file:
        tree = file["output"]
        track_x   = tree["trackPosX"].array()
        track_y   = tree["trackPosY"].array()
        track_z   = tree["trackPosZ"].array()
        track_pdg = tree["trackPDG"].array()
        track_ke  = tree["trackKE"].array()

    num_events = len(track_x)
    print(f"Loaded {num_events} events. Use ← → arrow keys to navigate.")

    state = {'event_index': 0}

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    plot_event(ax, track_x, track_y, track_z, track_pdg, track_ke, state['event_index'])

    def on_key(event):
        if event.key == 'right':
            state['event_index'] = min(state['event_index'] + 1, num_events - 1)
        elif event.key == 'left':
            state['event_index'] = max(state['event_index'] - 1, 0)
        else:
            return
        plot_event(ax, track_x, track_y, track_z, track_pdg, track_ke, state['event_index'])

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

if __name__ == "__main__":
    browse_events("../SimData/muon2_combined.root")