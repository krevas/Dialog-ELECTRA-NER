import streamlit as st
from bokeh.models.widgets.markups import Div

def create_explainer(color_dict, ent_dict):

    explainer = """<br>"""

    for ent_type in ent_dict:
        dark, light = color_dict[ent_dict[ent_type]]
        ent_html = f"""<b><span style="color: {dark}">{ent_type}   </span></b><br>"""
        explainer += ent_html

    return Div(text=explainer, width=500)


def produce_text_display(text, entity_list, color_dict):
    def style_wrapper(s, tag, tooltip=False):
        # Wraps a word that at least one model predicted to be an entity.
        dark, light = color_dict[tag]
        color = dark
        
        if tooltip:
            long_tag_names = {  # Define longer tag names for tooltip clarity
                "PS": "PERSON",
                "LC": "LOCATION",
                "OG": "ORGANIZATION",
                "DT": "Date",
                "TI":"Time"
            }
            html = f"""<span class="pred" style="background-color: {color}">
            <span class="tooltip">
            {s}
            <span class="tooltiptext" style="background-color: {color}">
                <b>{long_tag_names[tag]}</b>
            </span>
            </span>
            </span>"""
        else:  # Simply change the inline color of the predicted word
            html = f"""<span style="color: {color}; font-weight: bold">
            {s}</span>"""

        return html.replace("\n", "")

    if entity_list:
        output = ""
        output += text[0:entity_list[0][0]]
        end = entity_list[0][0]
        for entity in entity_list:
            output += text[end:entity[0]]
            output += style_wrapper(text[entity[0]:entity[1]], entity[2])
            end = entity[1]
        output += text[end::]
    else:
        output = text
    html_string = (
        """<div style="font-size: 18px; border-color: black">"""
        + output
        + "</div>"
    )

    return Div(text=html_string, width=700)