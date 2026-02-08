# My Agentic AI Experiment and Learning Project

Experiments with Gemini and xAI APIs using intelligent caching.

## Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager

### Install uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Or via Homebrew:**
```bash
brew install uv
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Quick Start

1. **Clone the repository**
```bash
   git clone https://github.com/TonyGregg/AgenticExperiments.git
   cd AgenticExperiments
```

2. **Install dependencies**
```bash
   # This installs all dependencies from pyproject.toml
   uv sync
```

3. **Set up environment variables**
```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env and add your API keys
   # GEMINI_API_KEY=your_gemini_key_here
   # XAI_API_KEY=your_xai_key_here
```

4. **Run the application**
```bash
   uv run python main.py
```

## Development

### Install with dev dependencies
```bash
uv sync --all-extras
```

### Run tests
```bash
uv run pytest
```

### Format code
```bash
uv run black src/
```

### Type checking
```bash
uv run mypy src/
```

## Project Structure
```
my-agentic-ai/
├── src/
│   ├── agents/          # AI agent implementations
│   ├── utils/           # Utilities (cache, config)
│   ├── models/          # Data models
│   └── services/        # API clients
├── tests/               # Unit tests
├── data/                # Cache and data files (gitignored)
├── main.py              # Entry point
├── pyproject.toml       # Dependencies and project config
└── uv.lock              # Locked dependency versions
```

## Adding New Dependencies
```bash
# Add runtime dependency
uv add package-name

# Add dev dependency
uv add --dev package-name
```

## License

MIT