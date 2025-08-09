import streamlit as st
import shap
import streamlit.components.v1 as components

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def render_shap_dashboard(text, explainer):
    shap_values = explainer([text])
    st_shap(shap.plots.text(shap_values[0]), height=300)