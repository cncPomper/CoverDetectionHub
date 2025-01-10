# gradio_app.py

import json
import gradio as gr

# Import the two Gradio interface functions
from utils.gradio_wrappers import gradio_cover_interface, gradio_test_interface


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
        gr.Dropdown(choices=["ByteCover", "CoverHunter", "Lyricover", "MFCC", "Spectral Centroid"], value="ByteCover", label="Choose CSI Model"),
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
        gr.Dropdown(choices=["ByteCover", "CoverHunter", "Lyricover", "MFCC", "Spectral Centroid"], value="ByteCover", label="Choose CSI Model"),
        gr.Dropdown(choices=["Covers80", "Covers80but10", "Injected Abracadabra"], value="Covers80", label="Choose Dataset"),
    ],
    outputs=[
        gr.Textbox(label="Summary Metrics")
    ],
    title="Model Testing on Dataset",
    description="Select a CSI model, dataset, and threshold to evaluate performance metrics such as mAP, P@10, and MR1."
)

demo = gr.TabbedInterface([app1, app2], ["Cover Song Identification", "Model Testing"])

if __name__ == "__main__":
    demo.launch(share=True)
