"""Run the entire capstone pipeline from the repository root."""
import subprocess,sys
for script in ['generate_data.py','part_a.py','part_b.py','part_c.py','part_d.py']:
    print(f'\n===== Running {script} =====',flush=True)
    subprocess.run([sys.executable,script],check=True)
print('\nCAPSTONE PIPELINE COMPLETED')
