# gradio_app.py
import json
import gradio as gr
from configs.gradio_config import public_dashboard
from utils.gradio_wrappers import (
    gradio_cover_interface,
    gradio_test_interface,
    gradio_generate_cover_interface
)

def parse_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

examples = parse_jsonl('examples.jsonl')


def parse_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

examples = parse_jsonl('examples.jsonl')

app1 = gr.Interface(
    fn=gradio_cover_interface,
    inputs=[
        gr.Audio(type="filepath", label="Query Song"),
        gr.Audio(type="filepath", label="Potential Cover Song"),
        gr.Dropdown(choices=["ByteCover", "CoverHunter", "Lyricover", "Lyricover Augmented", "MFCC", "Remove"], value="ByteCover", label="Choose CSI Model"),
        gr.Slider(minimum=0.5, maximum=1.0, step=0.001, value=0.99, label="Threshold")
    ],
    outputs=[
        gr.Textbox(label="Result"),
        gr.Textbox(label="Similarity Score")
    ],
    title="Cover Song Identification Hub",
    description="Upload two audio files to check if they are covers of each other. Select the desired CSI model and set the threshold.",
    examples=examples
)

app2 = gr.Interface(
    fn=gradio_test_interface,
    inputs=[
        gr.Dropdown(choices=["ByteCover", "CoverHunter", "Lyricover", "Lyricover Augmented", "MFCC", "Remove"], value="ByteCover", label="Choose CSI Model"),
        gr.Dropdown(choices=["Covers80", "Covers80but10", "Injected Abracadabra","DaTacos", "distracted_dataset", "distracted_dataset_reference"], value="Covers80", label="Choose Dataset"),
    ],
    outputs=[
        gr.Textbox(label="Summary Metrics")
    ],
    title="Model Testing on Dataset",
    description="Select a CSI model, dataset, and threshold to evaluate performance metrics such as mAP, P@10, and MR1."
)

app3 = gr.Interface(
    fn=gradio_generate_cover_interface,
    inputs=[
        gr.Audio(type="filepath", label="Input Song (melody + vocals?)"),
        gr.Slider(minimum=5, maximum=180, value=30, step=1, label="Cover Duration (seconds)"),
        gr.Radio(choices=["Instrumental", "With Lyrics"], value="With Lyrics", label="Cover Type"),
        gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.01, label="Timbre Shift Factor (alpha)"),
    ],
    outputs=[
        gr.Audio(type="filepath", label="Generated Cover")
    ],
    title="Cover Generation with Timbre Shifting",
    description=(
        "Upload a melodic clip (optionally with vocals).  \n"
        "- **Instrumental**: otrzymujesz tylko podkład.  \n"
        "- **With Lyrics**: rozdzielamy wokal, modyfikujemy barwę (alpha), i składamy ponownie."
    )
)

demo = gr.TabbedInterface(
    [app1, app2, app3],
    ["Cover Song Identification", "Model Testing", "Cover Generation"]
)

if __name__ == "__main__":
    demo.launch(share=public_dashboard)