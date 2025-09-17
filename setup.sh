# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install testing and documentation tools
pip install pytest sphinx sphinx_rtd_theme autograd
