import streamlit as st

def create_explainer(color_dict, ent_dict):

    explainer = """<b><span style="font-weight:bold">Type of entity</span></b><br>"""

    for ent_type in ent_dict:
        dark, light = color_dict[ent_dict[ent_type]]
        ent_html = f"""<b><span style="color: {dark}">{ent_type} &nbsp;&nbsp; </span></b>"""
        explainer += ent_html

    return explainer

def produce_text_display(text_list, entity_list, color_dict):
    def style_wrapper(s, tag, tooltip=False):
        dark, light = color_dict[tag]
        color = dark
        
        if tooltip:
            long_tag_names = {
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
        else:
            html = f"""<span style="color: {color};font-weight:bold">{s}</span>"""

        return html.replace("\n", "")
    
    output_html = []
    for text, entities in zip(text_list, entity_list):
        if entities:
            output = ""
            output += text[0:entities[0][0]]
            end = entities[0][0]
            for entity in entities:
                output += text[end:entity[0]]
                output += style_wrapper(text[entity[0]:entity[1]], entity[2], False)
                end = entity[1]
            output += text[end::]
        else:
            output = text
        output_html.append(output)

    html_string = (
            """<div style="font-size: 16px; border-color: black";display: flex; justify-content: center;>"""
        + ' '.join(output_html)
        + "</div>"
    )

    return html_string
