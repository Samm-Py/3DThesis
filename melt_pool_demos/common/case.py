"""Helpers for creating isolated, regenerable 3DThesis snapshot cases."""

from pathlib import Path


def ensure_snapshot_case(
    case_dir,
    *,
    name="TestSim",
    initial_temperature=1273.0,
    liquidus_temperature=1733.0,
    conductivity=26.6,
    heat_capacity=600.0,
    density=7451.0,
    threads=4,
):
    """Create the static files shared by snapshot-based optimization runs."""
    case = Path(case_dir)
    (case / "Data").mkdir(parents=True, exist_ok=True)

    files = {
        "ParamInput.txt": f"""Simulation
{{
\tName\t\t{name}
\tMode\t\tMode.txt
\tMaterial\tMaterial.txt
\tBeam\t\tBeam.txt
\tPath\t\tPath.txt
}}
Options
{{
\tDomain\t\tDomain.txt
\tOutput\t\tOutput.txt
\tSettings\tSettings.txt
}}
""",
        "Mode.txt": "Snapshots\n{\n\tScanFracs \t100\n\tTracking\tNone\n}\n",
        "Material.txt": f"""Constants
{{
\tT_0\t{initial_temperature}
\tT_L\t{liquidus_temperature}
\tk\t{conductivity}
\tc\t{heat_capacity}
\tp\t{density}
}}
""",
        "Output.txt": "Grid\n{\n\tx\t1\n\ty\t1\n\tz\t1\n}\n"
        "Temperature\n{\n\tT\t1\n\tT_hist\t0\n}\n",
        "Settings.txt": f"Compute\n{{\n\tMaxThreads\t{threads}\n}}\n",
    }
    for filename, content in files.items():
        (case / filename).write_text(content)

