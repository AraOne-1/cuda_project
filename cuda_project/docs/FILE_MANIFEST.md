# File manifest

| File | Purpose |
|---|---|
| `src/common.hpp` | Binary loader, optimized exact sequential CPU 1-NN, accuracy and result exports |
| `src/cpu_only.cpp` | CPU-only executable entry point (no NVIDIA needed) |
| `src/main.cu` | Two CUDA implementations, GPU timing, CPU/GPU cross-validation |
| `scripts/prepare_data.py` | Download and split public UCI Covertype dataset, reference-only scaling |
| `scripts/benchmark.py` | Benchmark scales and plot real runtime/speedup |
| `scripts/make_tiny_fixture.py` | Generate tiny handmade sanity dataset |
| `scripts/test_correctness.py` | Independent NumPy vs CPU correctness test |
| `scripts/fill_results_slide.py` | Populate slide 12 after authentic CUDA benchmarks |
| `scripts/make_presentation.py` | Re-create editable PowerPoint slides |
| `data/tiny.bin` | Small sanity dataset (not part of large benchmark) |
| `README.md` | Comprehensive methodology and build/run guide |
| `docs/RUN_GUIDE.pdf` | Submission-ready standalone how-to-run document |
| `docs/CUDA_Covertype_Presentation.pptx` | Editable 15-slide presentation |
| `docs/CUDA_Covertype_Presentation.pdf` | PDF viewing copy of the slides |
| `docs/PROJECT_REPORT.md` | Full algorithm/data/complexity report |
| `docs/DEMO_SCRIPT_TH.md` | Thai 15-minute presentation talk track and Q&A |
| `docs/SUBMISSION_CHECKLIST_TH.md` | Assignment compliance checklist |
| `docs/COLAB_GPU_TH.md` | Alternative GPU run instructions if local machine lacks NVIDIA |
| `results/README.md` | Where generated empirical results go |

**Not included:** full dataset binary (generated using `prepare_data.py`), CUDA benchmark results (must be measured on GPU), and GitHub repository URL (requires the student's GitHub account).
