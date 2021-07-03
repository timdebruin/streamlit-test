from typing import List

import altair as alt
import cv2
import larq_zoo as lz
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


def load_network(network_name: str) -> tf.keras.models.Model:
    net_class = getattr(lz.sota, network_name, None)
    if net_class is None:
        net_class = getattr(lz.literature, network_name, None)
    return net_class()


def resize_image(image: np.ndarray, desired_shape: tuple) -> np.ndarray:
    input_h, input_w = image.shape[:2]
    target_h, target_w = desired_shape[1:3]
    if not (input_h == target_h and input_w == target_w):
        image = image.astype(np.float32) / 255.0
        if input_h / input_w >= target_h / target_w:
            # If we're here then the current aspect ratio is 'too tall', so
            # we resize to perfectly fit the width.
            #     We can do this by setting `preserve_aspect_ratio=True`,
            # and using the correct target width with a huge target height
            # (since when preserving the aspect ratio the target dimensions
            # are maxima).
            resized_image = tf.image.resize(
                image,
                (int(1e6 * target_w), target_w),
                preserve_aspect_ratio=True,
            )
        else:
            # Or otherwise, resize to perfectly fit the height.
            resized_image = tf.image.resize(
                image,
                (target_h, int(1e6 * target_h)),
                preserve_aspect_ratio=True,
            )

        # Then crop to the target size (there will be no padding).
        image = (
            255
            * tf.image.resize_with_crop_or_pad(
                resized_image, target_h, target_w
            ).numpy()
        )

        image = image.astype(np.uint8)

    return image[np.newaxis, ...]


def decoded_prediction_plot(decoded_predictions: List[List[list]]) -> alt.BoxPlot:
    decoded_predictions = decoded_predictions[0]
    class_names = [p[1] for p in decoded_predictions]
    confidences = [float(p[2]) for p in decoded_predictions]
    return (
        alt.Chart(pd.DataFrame({"class": class_names, "confidence": confidences}))
        .mark_bar()
        .encode(x="confidence", y=alt.Y("class:N", sort="-x"))
    )


def main():
    st.title("Larq zoo models")

    modules = ["sota", "literature"]
    with st.sidebar:
        filters = st.multiselect("category", options=modules, default=modules)
    networks = []
    for filter in filters:
        networks.extend(getattr(lz, filter).__all__)

    chosen_network = st.selectbox("Choose network", options=["choose one"] + networks)

    if chosen_network != "choose one":
        st.write(chosen_network)
        if (
            "loaded_network" not in st.session_state
            or st.session_state.loaded_network_name != chosen_network
        ):
            with st.spinner(f"loading {chosen_network}"):
                st.session_state.loaded_network = load_network(chosen_network)
                st.session_state.loaded_network_name = chosen_network

        if chosen_network != "choose one":
            net = st.session_state.loaded_network
            uploaded_file = st.file_uploader("Choose an image")
            if uploaded_file is not None:
                image = cv2.imdecode(
                    np.frombuffer(uploaded_file.getvalue(), np.uint8), -1
                )
                if image.shape[-1] == 3:
                    # bgr to rgb
                    image = image[..., ::-1].copy()
                elif image.shape[-1] == 4:
                    image = image[..., :-1][..., ::-1].copy()

                image_resized = resize_image(image, net.input.shape)
                st.image(image_resized)

                prediction = net(-1 + image_resized / 127.5).numpy()
                decoded_predictions = lz.decode_predictions(prediction, top=1000)
                st.altair_chart(
                    decoded_prediction_plot(decoded_predictions),
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
