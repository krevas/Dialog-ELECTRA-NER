import streamlit as st

def create_explainer(color_dict, ent_dict):

    explainer = """<br> Entity : """

    for ent_type in ent_dict:
        dark, light = color_dict[ent_dict[ent_type]]
        ent_html = f"""<b><span style="color: {dark}">{ent_type}   </span></b>"""
        explainer += ent_html

    return explainer


def produce_text_display(text, entity_list, color_dict):
    def style_wrapper(s, tag, tooltip=False):
        # Wraps a word that at least one model predicted to be an entity.
        dark, light = color_dict[tag]
        color = dark
        
        if tooltip:
            long_tag_names = {  # Define longer tag names for tooltip clarity
                "PS": "인명",
                "LC": "장소",
                "OG": "기관",
                "DT": "날짜",
                "TI": "시간"
            }
            html = f"""<span class="pred" style="color: {color}">
            <span class="tooltip">
            {s}<span class="tooltiptext" style="color: {color}; font-size:11px"><b>[{long_tag_names[tag]}]</b>
            </span>
            </span>
            </span>"""
        else:  # Simply change the inline color of the predicted word
            html = f"""<span style="color: {color}">{s}</span>"""

        return html.replace("\n", "")

    if entity_list:
        output = ""
        output += text[0:entity_list[0][0]]
        end = entity_list[0][0]
        for entity in entity_list:
            output += text[end:entity[0]]
            output += style_wrapper(text[entity[0]:entity[1]], entity[2], False)
            end = entity[1]
        output += text[end::]
    else:
        output = text
    html_string = (
            """<div style="font-size: 16px; border-color: black";display: flex; justify-content: center;>"""
        + output
        + "</div>"
    )

    return html_string
