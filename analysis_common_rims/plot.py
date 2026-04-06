import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def parse_analysis_out(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    blocks = re.split(r'\n(?=\d+\.?\d*\n)', content)
    
    data = {}
    
    angle_pattern = re.compile(r'^(\d+\.?\d*)\s*$', re.MULTILINE)
    ric_pattern = re.compile(
        r'\s+\d+\s+(bond|angle|dihedral)\s+([\w\s]+?)\s{2,}([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)'
    )
    
    angles = angle_pattern.findall(content)
    jedi_blocks = re.split(r'^\d+\.?\d*\s*$', content, flags=re.MULTILINE)
    
    for i, (angle, block) in enumerate(zip(angles, jedi_blocks[1:])):
        angle = float(angle)
        rics = {}
        for match in ric_pattern.finditer(block):
            ric_type = match.group(1)
            indices = tuple(match.group(2).strip().split())
            percentage = float(match.group(4))
            energy = float(match.group(5))
            rics[indices] = {'type': ric_type, 'percentage': percentage, 'energy': energy}
        data[angle] = rics
    
    return data

def plot_ric_percentages(data):
    all_indices = set()
    for rics in data.values():
        all_indices.update(rics.keys())
    
    angles = sorted(data.keys())
    
    colors_bonds = plt.cm.Blues(np.linspace(0.4, 0.9, sum(1 for idx in all_indices 
                                if data[angles[0]].get(idx, {}).get('type') == 'bond')))
    colors_angles = plt.cm.Reds(np.linspace(0.4, 0.9, sum(1 for idx in all_indices 
                                if data[angles[0]].get(idx, {}).get('type') == 'angle')))
    colors_dihedrals = plt.cm.Greens(np.linspace(0.4, 0.9, sum(1 for idx in all_indices 
                                if data[angles[0]].get(idx, {}).get('type') == 'dihedral')))
    
    color_map = {}
    bi, ai, di = 0, 0, 0
    for idx in sorted(all_indices):
        ric_type = next((data[a][idx]['type'] for a in angles if idx in data[a]), None)
        if ric_type == 'bond':
            color_map[idx] = colors_bonds[bi]; bi += 1
        elif ric_type == 'angle':
            color_map[idx] = colors_angles[ai]; ai += 1
        elif ric_type == 'dihedral':
            color_map[idx] = colors_dihedrals[di]; di += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    
    legend_entries = []
    for idx in sorted(all_indices):
        ric_type = next((data[a][idx]['type'] for a in angles if idx in data[a]), None)
        percentages = [data[a][idx]['percentage'] if idx in data[a] else np.nan for a in angles]
        label = f"{ric_type}: {' – '.join(idx)}"
        line, = ax.plot(angles, percentages, marker='o', color=color_map[idx], linewidth=1.5, markersize=4)
        legend_entries.append((line, label))
    
    ax.set_xlabel('C–C–C Angle (°)', fontsize=12)
    ax.set_ylabel('Relative Strain Energy (%)', fontsize=12)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.tick_params(direction='in')
    
    handles, labels = zip(*legend_entries)
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=2, fontsize=12, frameon=False)
    
    plt.tight_layout()
    plt.savefig('jedi_strain.pdf', bbox_inches='tight')
    plt.show()

data = parse_analysis_out('analysis.out')
plot_ric_percentages(data)
