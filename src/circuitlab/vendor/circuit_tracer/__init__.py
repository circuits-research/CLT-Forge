import sys

from . import circuit_tracer as _circuit_tracer

# Make it importable as if it were a top-level package
sys.modules["circuit_tracer"] = _circuit_tracer
