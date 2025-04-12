# Python Virtual Environment Setup

A Python 3.11 virtual environment has been set up for this project.

## Automatic Activation

The environment is configured to activate automatically when you enter this directory, using `direnv`. 

### First-time setup:

1. Make sure direnv hooks are loaded in your shell:
   - For Zsh: `eval "$(direnv hook zsh)"` should be in your `~/.zshrc`
   - For Bash: `eval "$(direnv hook bash)"` should be in your `~/.bashrc`

2. Open a new terminal or run `source ~/.zshrc` (or `source ~/.bashrc` for bash)

3. When you `cd` into this directory, the environment will automatically activate

## Manual Activation

If automatic activation doesn't work, you can manually activate the environment:

```bash
# Option 1: Use the provided script
./activate_env.sh

# Option 2: Activate directly
source .venv/bin/activate
```

## Environment Details

- Python version: 3.11.11
- Virtual environment location: `./.venv/`

## Installing Packages

After activating the environment, you can install packages using pip:

```bash
pip install package_name
```

Or using uv (faster):

```bash
uv pip install package_name
```

## Deactivating the Environment

To deactivate the environment, simply run:

```bash
deactivate
```

Or leave the directory if using direnv.
