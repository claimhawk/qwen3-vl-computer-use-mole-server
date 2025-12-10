# MOLE Trainer Server - Development Progress

## Summary
- Development period: Nov 24 - Dec 5, 2025 (12 days)
- Total commits: 18

## Commit History by Feature

| Date | Feature | Work |
|------|---------|------|
| 2025-11-24 | Project Setup | Initial project setup |
| 2025-11-24 | Dataset Generation | Refactor routing dataset pipeline for Modal-based generation |
| 2025-11-24 | Cleanup | Remove unused qwenvl/ reference code |
| 2025-11-25 | Inference Pipeline | Clean up obsolete docs and plans, add stacked inference |
| 2025-12-01 | Documentation | Add README with contributing section |
| 2025-12-01 | Configuration | Add system prompt text file |
| 2025-12-01 | OCR Integration | Add Chandra OCR routing support |
| 2025-12-01 | OCR Integration | Add OCR/Chandra routing to unified routing dataset |
| 2025-12-01 | Licensing | Update license to Tylt proprietary (research use only) |
| 2025-12-02 | Multi-Adapter Routing | Add login-window, appointment, desktop, chart-screen adapters to routing |
| 2025-12-02 | Documentation | Add agents.md with commit guidelines and CLAUDE.md symlink |
| 2025-12-03 | Configuration | Refactor to use centralized config from adapters.yaml |
| 2025-12-04 | Dataset Generation | Refactor routing dataset generation to use YAML config |
| 2025-12-05 | Performance | Add parallel image caching and project tooling improvements |
| 2025-12-05 | Error Handling | Add better error logging for missing images in preprocessing |
| 2025-12-05 | Cleanup | Add Python bytecode and images to gitignore |
| 2025-12-05 | Training History | Add training history tracking for router training |
| 2025-12-05 | Training History | Add backfill_history.py script for router training |

## Feature Summary

### Project Setup (2025-11-24)
- Initial repository creation
- Basic project structure

### Dataset Generation (2025-11-24, 2025-12-04)
- Modal-based routing dataset pipeline
- YAML configuration for dataset generation
- Auto-discovery of latest datasets per adapter
- Configurable per-adapter sample counts with train/val/test splits

### Inference Pipeline (2025-11-25)
- Stacked inference: router -> task LoRA -> tool_call
- Routing accuracy evaluation
- Removed obsolete router head architecture docs

### OCR Integration (2025-12-01)
- Chandra OCR model integration
- OCR prompt templates with varied phrasings
- Unified OCR routing in main dataset generation

### Multi-Adapter Routing (2025-12-02)
- Added login-window adapter (label 6)
- Added appointment adapter (label 5)
- Added desktop adapter (label 4)
- Added chart-screen adapter (label 7)

### Configuration Management (2025-12-01, 2025-12-03)
- System prompt externalization
- Centralized adapters.yaml configuration
- SDK modal_compat helpers with fallbacks

### Performance Improvements (2025-12-05)
- Parallel image caching with ThreadPoolExecutor (8 workers, ~20-30x speedup)
- Better error logging for missing images

### Training History (2025-12-05)
- Persistent training metrics tracking
- Integration with shared claimhawk-training-history volume
- Records val_loss, eval_accuracy, cost, per-class accuracy
- Backfill script for historical runs

### Documentation & Licensing (2025-12-01, 2025-12-02)
- README with contributing guidelines
- Tylt LLC proprietary license
- agents.md with commit guidelines
- CLAUDE.md integration

---

## AI-Assisted Development

Built by 1 developer + AI (Claude Code). 18 commits in 12 days.

### Cost Comparison

- **Traditional:** 2 ML engineers @ $150k/yr for 3-4 weeks = **$17-23k**
- **Actual:** 1 developer + AI, 12 days = **$1-2k**
- **Savings: ~90%**

Router training infrastructure with parallel preprocessing and multi-adapter support.
