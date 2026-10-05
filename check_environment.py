"""Environment diagnostic check for Opportunity_HAR.

Reports Python version, PyTorch version, CUDA availability, GPU device details,
CPU architecture, and available system RAM.
"""

import sys
import platform
import torch


def check_environment() -> dict:
    """Collect and display hardware and runtime software environment info."""
    print("=" * 60)
    print(" Opportunity_HAR: Runtime Environment Verification")
    print("=" * 60)

    # Python & OS
    py_ver = sys.version.replace("\n", " ")
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    cpu_info = platform.processor() or "Unknown CPU"

    print(f"Python Version    : {py_ver}")
    print(f"Operating System  : {os_info}")
    print(f"CPU Architecture  : {cpu_info}")

    # RAM (try psutil, fallback to os/ctypes if not available)
    ram_gb = "N/A"
    try:
        import psutil
        ram_gb = f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB"
    except ImportError:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', c_ulonglong),
                    ('ullAvailPhys', c_ulonglong),
                    ('ullTotalPageFile', c_ulonglong),
                    ('ullAvailPageFile', c_ulonglong),
                    ('ullTotalVirtual', c_ulonglong),
                    ('ullAvailVirtual', c_ulonglong),
                    ('ullAvailExtendedVirtual', c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                ram_gb = f"{round(stat.ullTotalPhys / (1024**3), 2)} GB"
        except Exception:
            ram_gb = "Unable to determine"

    print(f"System RAM        : {ram_gb}")

    # PyTorch & CUDA
    torch_ver = torch.__version__
    cuda_avail = torch.cuda.is_available()
    cuda_ver = torch.version.cuda if cuda_avail else "N/A"

    print(f"PyTorch Version   : {torch_ver}")
    print(f"CUDA Available    : {cuda_avail}")
    print(f"CUDA Build Version: {cuda_ver}")

    gpu_details = []
    if cuda_avail:
        device_count = torch.cuda.device_count()
        print(f"CUDA Devices Found: {device_count}")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            mem_gb = round(props.total_memory / (1024**3), 2)
            desc = f"GPU {i}: {props.name} ({mem_gb} GB, Compute {props.major}.{props.minor})"
            print(f"  -> {desc}")
            gpu_details.append(desc)
    else:
        print("Compute Device    : CPU fallback active (CUDA not available)")

    print("=" * 60)

    return {
        "python_version": py_ver,
        "os": os_info,
        "cpu": cpu_info,
        "ram": ram_gb,
        "torch_version": torch_ver,
        "cuda_available": cuda_avail,
        "gpu_details": gpu_details,
    }


if __name__ == "__main__":
    check_environment()
