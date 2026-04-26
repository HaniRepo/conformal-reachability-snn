import os
import numpy as np


def load_npz_file(filepath: str) -> np.ndarray:
    """
    Load a .npz file and extract the DE signal as a 1D numpy array.
    Handles regular arrays and object arrays safely.
    """
    data = np.load(filepath, allow_pickle=True)

    if "DE" not in data:
        raise ValueError(f"No 'DE' key found in file: {filepath}")

    signal = data["DE"]

    if signal.dtype == object:
        if signal.size == 1:
            signal = signal.item()
        else:
            signal = np.concatenate([np.ravel(x) for x in signal])

    signal = np.asarray(signal)
    signal = signal.squeeze()
    signal = signal.ravel()

    if signal.size == 0:
        raise ValueError(f"Empty DE signal in file: {filepath}")

    return signal


def load_normal_and_fault(data_path: str, rpm_folder: str = "1797 RPM") -> tuple[np.ndarray, np.ndarray]:
    """
    Load one normal file and one inner-race fault file from the given RPM folder.
    Kept for backward compatibility.
    """
    folder = os.path.join(data_path, rpm_folder)

    if not os.path.isdir(folder):
        raise ValueError(f"Folder does not exist: {folder}")

    files = [f for f in os.listdir(folder) if f.endswith(".npz")]

    normal_file = None
    fault_file = None

    for f in files:
        if "Normal" in f and normal_file is None:
            normal_file = f
        if "IR" in f and "DE" in f and fault_file is None:
            fault_file = f

    if normal_file is None:
        raise ValueError(f"Could not find a normal file in: {folder}")

    if fault_file is None:
        raise ValueError(f"Could not find an IR drive-end fault file in: {folder}")

    normal_path = os.path.join(folder, normal_file)
    fault_path = os.path.join(folder, fault_file)

    print(f"Using normal file: {normal_file}")
    print(f"Using fault file:  {fault_file}")

    normal_signal = load_npz_file(normal_path)
    fault_signal = load_npz_file(fault_path)

    return normal_signal, fault_signal


def load_severity_signals(
    data_path: str,
    rpm_folder: str = "1797 RPM",
    fault_type: str = "IR",
    severities: tuple[int, int, int] = (7, 14, 21),
    end_tag: str = "DE12",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one normal signal and three increasing fault-severity signals
    from the given RPM folder.

    Default progression:
        healthy -> IR_7 -> IR_14 -> IR_21

    Returns:
        normal_signal, mild_signal, medium_signal, severe_signal
    """
    folder = os.path.join(data_path, rpm_folder)

    if not os.path.isdir(folder):
        raise ValueError(f"Folder does not exist: {folder}")

    files = [f for f in os.listdir(folder) if f.endswith(".npz")]

    normal_file = None
    severity_files = {}

    for f in files:
        if "Normal" in f and normal_file is None:
            normal_file = f

        for sev in severities:
            target = f"{fault_type}_{sev}_{end_tag}"
            if target in f:
                severity_files[sev] = f

    if normal_file is None:
        raise ValueError(f"Could not find a normal file in: {folder}")

    missing = [sev for sev in severities if sev not in severity_files]
    if missing:
        raise ValueError(
            f"Could not find required severity files for {fault_type} {missing} in: {folder}"
        )

    normal_path = os.path.join(folder, normal_file)
    mild_path = os.path.join(folder, severity_files[severities[0]])
    medium_path = os.path.join(folder, severity_files[severities[1]])
    severe_path = os.path.join(folder, severity_files[severities[2]])

    print(f"Using normal file:  {normal_file}")
    print(f"Using mild file:    {severity_files[severities[0]]}")
    print(f"Using medium file:  {severity_files[severities[1]]}")
    print(f"Using severe file:  {severity_files[severities[2]]}")

    normal_signal = load_npz_file(normal_path)
    mild_signal = load_npz_file(mild_path)
    medium_signal = load_npz_file(medium_path)
    severe_signal = load_npz_file(severe_path)

    return normal_signal, mild_signal, medium_signal, severe_signal