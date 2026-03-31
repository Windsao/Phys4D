from uipc import Logger
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def create_profiler_heatmap(
    timer_data, output_path=None, max_depth=999, include_other=True
):
    '''
    Create a hierarchical heatmap chart from profiling timer data.

    Usage:
        create_profiler_heatmap(uipc.Timer.report_as_json())

    :param timer_data: uipc timer data in JSON format, as returned by uipc.Timer.report_as_json()
    :param output_path: Path to save the output chart image. If None, the chart will be displayed but not saved.
    :param max_depth: Maximum depth of the hierarchy to include in the heatmap.
    :param include_other: Whether to include "other" categories in the heatmap.
    '''

    level_data = {}
    node_angles = {}

    legend_items = []
    total_duration = (
        timer_data['children'][0].get('duration', 0) if timer_data['children'][0] else 0
    )
    frame_count = (
        timer_data['children'][0].get('count', 0) if timer_data['children'][0] else 0
    )

    if total_duration == 0:
        Logger.warn('Total duration is zero, cannot create heatmap')
        return

    def collect_level_data(node, depth=0, parent_name='root'):
        '''Recursively collect hierarchical data'''
        if depth > max_depth:
            return

        name = node.get('name', 'Unknown')
        duration = node.get('duration', 0)
        count = node.get('count', 0)


        level_data.setdefault(depth, {})[name] = {
            'name': name,
            'full_name': f'{parent_name} -> {name}' if depth > 0 else name,
            'duration': duration,
            'percentage': 0,
            'parent': parent_name if depth > 0 else None,
            'count': count,
        }


        if 'children' in node:
            for child in node['children']:
                collect_level_data(child, depth + 1, name)


    collect_level_data(timer_data)

    if not level_data:
        Logger.warn('No valid hierarchical data collected')
        return


    plt.figure(figsize=(15, 10))
    ring_width = 0.3

    max_observed_depth = max(level_data.keys()) if level_data else 0
    if include_other:
        for depth in sorted(level_data.keys()):
            if depth == max_observed_depth:
                continue
            for parent_name, parent_data in level_data[depth].items():
                if 'other' in parent_name:
                    continue
                parent_duration = parent_data['duration']
                child_total_duration = 0
                child_count = 0
                if depth + 1 in level_data:
                    for child_data in level_data[depth + 1].values():
                        if (
                            child_data['parent'] == parent_name
                            and 'other' not in parent_data['name']
                        ):
                            child_total_duration += child_data['duration']
                            child_count += 1
                if child_count == 0:

                    continue

                remaining_time = parent_duration - child_total_duration

                if remaining_time > 0.000001:
                    remaining_percentage = (remaining_time / total_duration) * 100

                    other_node = {
                        'name': 'Other',
                        'full_name': f'{parent_name} -> Other',
                        'duration': remaining_time,
                        'percentage': remaining_percentage,
                        'parent': parent_name,
                        'count': 1,
                        'is_other': True,
                    }

                    other_key = f'{parent_name}_other'
                    level_data.setdefault(depth + 1, {})[other_key] = other_node

    for depth, nodes in level_data.items():
        if depth == 0:
            continue

        inner_radius = 0.25 + (max_observed_depth - depth) * ring_width
        if depth == 1:
            sorted_items = nodes.values()
            sizes = [item['duration'] for item in sorted_items]
            for item in sorted_items:
                item['percentage'] = (item['duration'] / total_duration) * 100

            wedges, _ = plt.pie(
                sizes,
                labels=nodes,
                radius=inner_radius + ring_width,
                startangle=90,
                counterclock=True,
                autopct=None,
                wedgeprops={
                    'width': ring_width,
                    'edgecolor': 'white',
                    'linewidth': 0.5,
                },
            )

            node_angles[1] = {}
            for i, (wedge, item) in enumerate(zip(wedges, sorted_items)):
                node_name = item['name']
                node_angles[1][node_name] = (wedge.theta1, wedge.theta2)

                angle_size = wedge.theta2 - wedge.theta1
                if not item.get('is_other', False):
                    ang = (wedge.theta2 + wedge.theta1) / 2
                    ang_rad = np.deg2rad(ang)

                    r = inner_radius + ring_width / 2
                    x = r * np.cos(ang_rad)
                    y = r * np.sin(ang_rad)
                plt.text(
                    x,
                    y,
                    f'{item["percentage"]:.2f}%',
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='white',
                    fontweight='bold',
                )
                legend_items.append(
                    (
                        item['full_name'],
                        item['percentage'],
                        wedge.get_facecolor(),
                    )
                )
        else:

            prev_level = level_data.get(depth - 1, {})
            parent_order = list(prev_level.keys())

            parent_groups = {p: [] for p in parent_order}
            for item in nodes.values():
                p = item['parent']
                if p in parent_groups:
                    parent_groups[p].append(item)
            sorted_items = []
            for p in parent_order:
                group = parent_groups.get(p, [])
                group_sorted = group
                sorted_items.extend(group_sorted)

            parent_groups = {}
            for item in sorted_items:
                parent = item['parent']
                if parent not in parent_groups:
                    parent_groups[parent] = []
                parent_groups[parent].append(item)


            node_angles[depth] = {}

            for parent, children in parent_groups.items():
                if parent not in level_data.get(depth - 1, {}):
                    continue
                for item in children:
                    item['percentage'] = (item['duration'] / total_duration) * 100
                child_sizes = [item['duration'] for item in children]

                if parent in node_angles.get(depth - 1, {}):
                    parent_start, parent_end = node_angles[depth - 1][parent]

                    parent_angle_size = parent_end - parent_start

                    wedges, _ = plt.pie(
                        child_sizes,
                        labels=None,
                        radius=inner_radius + ring_width,
                        startangle=90,
                        counterclock=True,
                        autopct=None,
                        wedgeprops={
                            'width': ring_width,
                            'edgecolor': 'white',
                            'linewidth': 0.5,
                        },
                    )


                    orig_angles = []
                    for wedge in wedges:
                        orig_angles.append((wedge.theta1, wedge.theta2))


                    total_child_duration = sum(child_sizes)


                    current_angle = parent_start


                    for i, (wedge, child_size) in enumerate(zip(wedges, child_sizes)):

                        angle_proportion = child_size / total_child_duration
                        angle_range = parent_angle_size * angle_proportion


                        new_theta1 = current_angle
                        new_theta2 = current_angle + angle_range


                        current_angle = new_theta2


                        wedge.set_theta1(new_theta1)
                        wedge.set_theta2(new_theta2)


                        if i < len(children):
                            children[i]['mapped_angles'] = (new_theta1, new_theta2)
                else:

                    wedges, _ = plt.pie(
                        child_sizes,
                        labels=None,
                        radius=inner_radius + ring_width,
                        startangle=0,
                        counterclock=True,
                        autopct=None,
                        wedgeprops={
                            'width': ring_width,
                            'edgecolor': 'white',
                            'linewidth': 0.5,
                        },
                    )

                for i, (wedge, item) in enumerate(zip(wedges, children)):
                    node_angles[depth][item['name']] = (wedge.theta1, wedge.theta2)

                    angle_size = wedge.theta2 - wedge.theta1
                    if not item.get('is_other', False):
                        ang = (wedge.theta2 + wedge.theta1) / 2
                        ang_rad = np.deg2rad(ang)

                        r = inner_radius + ring_width / 2
                        x = r * np.cos(ang_rad)
                        y = r * np.sin(ang_rad)
                        if angle_size > 5:
                            plt.text(
                                x,
                                y,
                                f'{item["percentage"]:.2f}%',
                                ha='center',
                                va='center',
                                fontsize=10,
                                color='white',
                                fontweight='bold',
                            )
                        legend_items.append(
                            (
                                item['full_name'],
                                item['percentage'],
                                wedge.get_facecolor(),
                            )
                        )


    plt.text(
        0,
        0,
        f'{total_duration:.3f}s',
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.95),
    )

    plt.axis('equal')

    legend_items.sort(key=lambda x: x[1], reverse=True)

    legend_patches = []
    for name, percentage, color in legend_items:
        label = f'{name} ({percentage:.2f}%)'
        patch = mpatches.Patch(label=label, color=color)
        legend_patches.append(patch)


    plt.legend(
        handles=legend_patches,
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=True,
        title=f'Total time:{total_duration:.3f}s/{frame_count} Frames',
    )

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.5)
    plt.tight_layout()
    plt.show()
