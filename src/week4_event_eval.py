"""
Compatibility wrapper for src.event_eval.
All canonical event evaluation and parameter grid search logic resides in src/event_eval.py.
"""

from src.event_eval import (
    Event,
    FrameDetection,
    evaluate,
    export_events_csv,
    export_grid_search_csv,
    export_json,
    extract_frame_level_alarms,
    frame_detections_to_events,
    grid_search,
    load_events,
    load_frame_detections,
    load_jsonl,
    main,
    merge_frame_alarms_to_events,
    select_best_operating_point,
)

__all__ = [
    "Event",
    "FrameDetection",
    "load_jsonl",
    "load_events",
    "load_frame_detections",
    "extract_frame_level_alarms",
    "merge_frame_alarms_to_events",
    "frame_detections_to_events",
    "evaluate",
    "grid_search",
    "select_best_operating_point",
    "export_json",
    "export_grid_search_csv",
    "export_events_csv",
    "main",
]

if __name__ == "__main__":
    main()
