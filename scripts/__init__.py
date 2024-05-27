import plotly.io as pio
import os

if os.getenv("PLOTLY_PDF"):
    pio.renderers.default = "pdf"
    pio.renderers.render_on_display = False
