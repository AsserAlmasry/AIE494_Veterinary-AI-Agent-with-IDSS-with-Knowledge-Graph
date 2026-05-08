import sys
import numpy

def apply_numpy_hack():
    # Compatibility hack for older/newer pickles saved with different numpy versions
    if not hasattr(numpy, "_core"):
        # We are on an older numpy (1.x), but pickle wants 2.x (or vice versa)
        try:
            import numpy.core as core
            sys.modules["numpy._core"] = core
            if hasattr(core, "numeric"):
                sys.modules["numpy._core.numeric"] = core.numeric
            else:
                import numpy.core.numeric as numeric
                sys.modules["numpy._core.numeric"] = numeric
        except (ImportError, AttributeError):
            pass
    elif not hasattr(numpy, "core"):
        # We are on a newer numpy (2.x), but pickle wants 1.x
        try:
            import numpy._core as core
            sys.modules["numpy.core"] = core
            if hasattr(core, "numeric"):
                sys.modules["numpy.core.numeric"] = core.numeric
        except (ImportError, AttributeError):
            pass

# Apply immediately on import
apply_numpy_hack()
