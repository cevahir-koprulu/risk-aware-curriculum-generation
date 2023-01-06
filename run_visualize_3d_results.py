import subprocess

filepath = r".\misc\visualize_3d_results.py"
arguments = [
            ['--algorithms', 'self_paced', 'self_paced_with_cem', 'default', 'default_with_cem', '--env', 'point_mass_2d', '--seeds', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '--eval_type', '0', '--result_type', '0', '--view_angle', '-90', '90'],
            ['--algorithms', 'self_paced', 'self_paced_with_cem', 'default', 'default_with_cem', '--env', 'point_mass_2d', '--seeds', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '--eval_type', '0', '--result_type', '1', '--view_angle', '-90', '90'],
            ['--algorithms', 'self_paced', 'self_paced_with_cem', 'default', 'default_with_cem', '--env', 'point_mass_2d', '--seeds', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '--eval_type', '3', '--result_type', '0', '--view_angle', '-90', '90'],
            ['--algorithms', 'self_paced', 'self_paced_with_cem', 'default', 'default_with_cem', '--env', 'point_mass_2d', '--seeds', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '--eval_type', '3', '--result_type', '1', '--view_angle', '-90', '90'],
]


for args in arguments:
    subprocess.call(["python", filepath] + [arg for arg in args])


filepath2 = r".\misc\visualize_3d_results_in_gif.py"
arguments2 = [
            ['--algorithms', 'self_paced', 'self_paced_with_cem', 'default', 'default_with_cem', '--env', 'point_mass_2d', '--eval_type', '0', '--result_type', '0', '--view_angle', '-90', '90'],
            ['--algorithms', 'self_paced', 'self_paced_with_cem', 'default', 'default_with_cem', '--env', 'point_mass_2d', '--eval_type', '0', '--result_type', '1', '--view_angle', '-90', '90'],
            ['--algorithms', 'self_paced', 'self_paced_with_cem', 'default', 'default_with_cem', '--env', 'point_mass_2d', '--eval_type', '3', '--result_type', '0', '--view_angle', '-90', '90'],
            ['--algorithms', 'self_paced', 'self_paced_with_cem', 'default', 'default_with_cem', '--env', 'point_mass_2d', '--eval_type', '3', '--result_type', '1', '--view_angle', '-90', '90'],

]

for args in arguments2:
    subprocess.call(["python", filepath2] + [arg for arg in args])