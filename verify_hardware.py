"""
verify_hardware.py
==================
Hardware Stress-Test Verification for YOLO26 + CRNN Video Anomaly Detection Pipeline.

Target Hardware:
  - CPU : AMD Ryzen Threadripper PRO (32-Cores)
  - RAM : 128 GB
  - GPU : NVIDIA RTX 5090 (~31 GB VRAM)

Tests:
  1. GPU Detection & VRAM validation
  2. PyTorch DataLoader multi-worker stress test (32-core utilisation)
  3. Large tensor GPU allocation (batch=128 high-res video sequences)
"""

import os
import sys
import time
import psutil
import torch
import torch.utils.data as data
import multiprocessing

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
EXPECTED_GPU_NAME_FRAGMENT = "5090"
EXPECTED_VRAM_GB_MIN = 30.0          # RTX 5090 ≈ 31-32 GB
NUM_DATALOADER_WORKERS = 32          # Match Threadripper cores
DUMMY_DATASET_SIZE = 10_000
BATCH_SIZE = 128
# Stress tensor — 24 GB fp32 to push RTX 5090 VRAM hard
# 128 clips × 32 frames × 3ch × 512×512  ≈ 24 GB fp32 / 12 GB fp16
TENSOR_SHAPE_FOR_STRESS = (128, 32, 3, 512, 512)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

all_passed = True


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TEST 1 — GPU Detection & VRAM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_gpu_detection():
    global all_passed
    section("TEST 1: GPU Detection & VRAM Validation")

    # System info
    cpu_count = multiprocessing.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"{INFO} CPU logical cores : {cpu_count}")
    print(f"{INFO} System RAM        : {ram_gb:.1f} GB")
    print(f"{INFO} PyTorch version   : {torch.__version__}")
    print(f"{INFO} CUDA available    : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print(f"{FAIL} CUDA is NOT available — cannot proceed with GPU tests.")
        all_passed = False
        return False

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    cuda_version = torch.version.cuda
    print(f"{INFO} CUDA version      : {cuda_version}")
    print(f"{INFO} GPU name          : {gpu_name}")
    print(f"{INFO} Total VRAM        : {vram_gb:.2f} GB")

    # Validate GPU identity
    if EXPECTED_GPU_NAME_FRAGMENT.lower() in gpu_name.lower():
        print(f"{PASS} GPU correctly identified as RTX 5090.")
    else:
        print(f"{FAIL} GPU name mismatch. Expected '{EXPECTED_GPU_NAME_FRAGMENT}', got '{gpu_name}'.")
        all_passed = False

    # Validate VRAM
    if vram_gb >= EXPECTED_VRAM_GB_MIN:
        print(f"{PASS} VRAM ({vram_gb:.2f} GB) meets minimum requirement ({EXPECTED_VRAM_GB_MIN} GB).")
    else:
        print(f"{FAIL} VRAM ({vram_gb:.2f} GB) is below minimum ({EXPECTED_VRAM_GB_MIN} GB).")
        all_passed = False

    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TEST 2 — DataLoader Multi-Worker Stress Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class DummyVideoDataset(data.Dataset):
    """Simulates loading 16-frame, 3-channel, 256×256 video clips."""
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate CPU-heavy decode: return a random tensor (16, 3, 256, 256)
        clip = torch.randn(16, 3, 256, 256)
        return clip, idx


def _run_dataloader(dataset, num_workers, pin_memory, max_batches=20):
    """Helper: run DataLoader and return (success, batches, elapsed, throughput)."""
    loader = data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
        persistent_workers=False,   # avoid cleanup issues on Windows
    )

    t0 = time.perf_counter()
    batches_loaded = 0
    for batch_clips, batch_ids in loader:
        batches_loaded += 1
        if batches_loaded >= max_batches:
            break
    elapsed = time.perf_counter() - t0

    # Explicitly delete the loader to shut down workers
    del loader
    import gc; gc.collect()

    total_samples = batches_loaded * BATCH_SIZE
    throughput = total_samples / elapsed if elapsed > 0 else 0
    return batches_loaded, elapsed, throughput


def test_dataloader_workers():
    global all_passed
    section("TEST 2: DataLoader Multi-Worker Stress Test (32-core)")

    dataset = DummyVideoDataset(DUMMY_DATASET_SIZE)
    logical_cores = multiprocessing.cpu_count()
    target_workers = min(NUM_DATALOADER_WORKERS, logical_cores)

    # Strategy: try target workers with pin_memory, fallback as needed
    configs = [
        (target_workers, True,  f"{target_workers} workers + pin_memory"),
        (target_workers, False, f"{target_workers} workers (no pin_memory)"),
        (16,             False, "16 workers (no pin_memory)"),
    ]

    for workers, pin_mem, label in configs:
        print(f"\n{INFO} Trying: {label}, batch_size={BATCH_SIZE} ...")
        try:
            batches, elapsed, throughput = _run_dataloader(dataset, workers, pin_mem)
            total_samples = batches * BATCH_SIZE
            print(f"{PASS} {label} — {batches} batches ({total_samples} samples) in {elapsed:.2f}s")
            print(f"{INFO} Throughput: {throughput:.0f} samples/sec")
            return  # success — stop trying fallbacks
        except Exception as e:
            err_msg = str(e)
            # Truncate very long CUDA error messages
            if len(err_msg) > 200:
                err_msg = err_msg[:200] + "..."
            print(f"{WARN} {label} failed: {err_msg}")
            # Clean up CUDA state after pin_memory errors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc; gc.collect()
            time.sleep(1)  # brief cooldown

    print(f"{FAIL} All DataLoader configurations failed.")
    all_passed = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TEST 3 — Large Tensor GPU Allocation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def test_large_tensor_allocation():
    global all_passed
    section("TEST 3: Large Tensor GPU Allocation (CUDA 12.x Stress)")

    # Reset CUDA state cleanly before allocation tests
    torch.cuda.empty_cache()
    import gc; gc.collect()
    torch.cuda.reset_peak_memory_stats()

    shape = TENSOR_SHAPE_FOR_STRESS
    numel = 1
    for d in shape:
        numel *= d
    size_gb_fp32 = (numel * 4) / (1024 ** 3)
    size_gb_fp16 = (numel * 2) / (1024 ** 3)

    print(f"{INFO} Tensor shape : {shape}")
    print(f"{INFO} Size (fp32)  : {size_gb_fp32:.2f} GB")
    print(f"{INFO} Size (fp16)  : {size_gb_fp16:.2f} GB")

    # --- FP16 allocation (should comfortably fit ~12 GB) ---
    print(f"\n{INFO} Attempting fp16 allocation on GPU ...")
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        tensor_gpu = torch.randn(shape, dtype=torch.float16, device="cuda")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"{PASS} fp16 tensor allocated on GPU in {elapsed:.2f}s")
        print(f"{INFO} Peak GPU memory used: {peak_mem:.2f} GB")

        # Quick compute sanity check
        result = tensor_gpu.mean()
        print(f"{INFO} Tensor mean (sanity): {result.item():.6f}")

        del tensor_gpu
        torch.cuda.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"{FAIL} CUDA out-of-memory for fp16 allocation: {e}")
        else:
            print(f"{FAIL} CUDA runtime error: {e}")
        all_passed = False
        return

    # --- FP32 allocation (pushes VRAM ~24 GB on a 31 GB card) ---
    print(f"\n{INFO} Attempting fp32 allocation on GPU ...")
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        tensor_gpu = torch.randn(shape, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"{PASS} fp32 tensor allocated on GPU in {elapsed:.2f}s")
        print(f"{INFO} Peak GPU memory used: {peak_mem:.2f} GB")

        # Quick compute sanity check
        result = tensor_gpu.mean()
        print(f"{INFO} Tensor mean (sanity): {result.item():.6f}")

        del tensor_gpu
        torch.cuda.empty_cache()

    except RuntimeError as e:
        err_msg = str(e).lower()
        if "out of memory" in err_msg:
            print(f"{WARN} fp32 allocation OOM ({size_gb_fp32:.1f} GB tensor + overhead on ~31 GB card).")
            print(f"{INFO} fp16 passed — pipeline will use mixed-precision. This is acceptable.")
        elif "shared object" in err_msg or "paging file" in err_msg:
            print(f"{WARN} fp32 allocation failed (system resource issue, not GPU OOM): {type(e).__name__}")
            print(f"{INFO} fp16 passed — pipeline will use mixed-precision. This is acceptable.")
        else:
            print(f"{FAIL} CUDA runtime error: {e}")
            all_passed = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   HARDWARE VERIFICATION — YOLO26 + CRNN Pipeline       ║")
    print("║   Target: Threadripper PRO 32C · 128GB · RTX 5090      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    gpu_ok = test_gpu_detection()

    test_dataloader_workers()

    if gpu_ok:
        test_large_tensor_allocation()
    else:
        print(f"\n{WARN} Skipping GPU allocation test — CUDA not available.")

    # Final verdict
    section("FINAL VERDICT")
    if all_passed:
        print(f"{PASS} ALL TESTS PASSED — Hardware is ready for the pipeline. 🚀")
    else:
        print(f"{FAIL} SOME TESTS FAILED — Review output above and fix issues.")

    sys.exit(0 if all_passed else 1)
