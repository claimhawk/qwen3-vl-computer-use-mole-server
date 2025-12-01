# MOLE Training Server

Distributed training infrastructure for Mixture of LoRA Experts (MOLE) models.

## Overview

This server coordinates distributed fine-tuning of vision-language models using the MOLE architecture, enabling efficient multi-expert training across GPU clusters.

## Setup

```bash
pip install -r requirements.txt
```

## Structure

- `modal/` - Modal cloud deployment functions
- `config/` - Training configurations
- `scripts/` - Utility scripts
- `routing-generator/` - Expert routing data generation
- `utils/` - Shared utilities
- `docs/` - Architecture and reference documentation

## Usage

See `docs/` for detailed usage instructions and architecture notes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes:
   - Generalize hardcoded values rather than replacing them with your own
   - Add tests for new functionality
   - Ensure all quality checks pass
4. Submit a pull request

**Code quality requirements:**
- Lexical complexity checks
- Syntax linting
- Code formatting
- Copyright headers

See `CODE_QUALITY.md` for detailed standards.

AI-assisted code is welcome provided it includes tests and passes all checks.

## License

Copyright (c) 2025 Tylt LLC. All rights reserved.
