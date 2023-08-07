from argparse import ArgumentParser

import altair.vega.v5 as vg


def visualize_in_html(vis_spec_path, output_file="index"):
    vis_spec = open(vis_spec_path, "r").read()

    chart = vg.Vega(vis_spec)

    chart.save(f"{output_file}.html")


if __name__ == "__main__":

    visualize_in_html("./visualization_spec.vg.json")