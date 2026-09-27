"""Built-in Ultralytics Streamlit inference UI with custom pre-selected classes."""

import sys

from ultralytics.cfg import parse_key_value_pair
from ultralytics.solutions.streamlit_inference import Inference

DEFAULT_CLASSES = ["person", "spoon", "cell phone", "cup"]


class App(Inference):
    def configure(self):
        # Upstream hardcodes default=class_names[:3] for the "Classes" multiselect; swap only that.
        sidebar = self.st.sidebar
        multiselect = type(sidebar).multiselect

        def patched(label, options, *args, **kwargs):
            if label == "Classes":
                kwargs["default"] = [c for c in DEFAULT_CLASSES if c in options]
            return multiselect(sidebar, label, options, *args, **kwargs)

        sidebar.multiselect = patched
        try:
            super().configure()
        finally:
            del sidebar.multiselect


if __name__ == "__main__":
    App(**dict(parse_key_value_pair(arg) for arg in sys.argv[1:])).inference()
