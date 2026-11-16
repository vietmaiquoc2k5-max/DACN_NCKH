"""Small bi-directional Streamlit component for camera location selection."""
from pathlib import Path
from typing import Any, Optional


def render_camera_location_map(latitude: Optional[float], longitude: Optional[float],
                               location_name: str = "", key: str = "camera_location_map") -> Any:
    import streamlit.components.v1 as components

    component = components.declare_component(
        "firewatch_camera_location",
        path=str(Path(__file__).resolve().parent.parent / "components" / "camera_location"),
    )
    return component(latitude=latitude, longitude=longitude, location_name=location_name,
                     key=key, default=None)
