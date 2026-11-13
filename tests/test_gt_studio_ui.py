"""Execute Streamlit login, blind source frame navigation and marking on synthetic media."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from gt_studio_fixtures import StudioCase


class TestStudioUI(StudioCase):
    def settle(self, app):
        deadline = time.monotonic() + 15
        while any("PREPARING_SOURCE" in item.value for item in app.info):
            if time.monotonic() > deadline:
                self.fail("Background source preparation exceeded fixture timeout")
            time.sleep(.05)
            app.run()

    def test_login_frame_seek_mark_and_save_draft(self):
        config = self.private / "ui_test_settings.json"
        config.write_text(json.dumps({"repo_root": str(self.root), "private_dir": str(self.private)}))
        entry = Path(__file__).resolve().parent.parent / "ground_truth_studio/app.py"
        with patch.dict("os.environ", {"GT_STUDIO_CONFIG": str(config), "GT_STUDIO_PRIVATE_DIR": str(self.private)}):
            app = AppTest.from_file(entry, default_timeout=60).run()
            self.assertFalse(app.exception)
            app.text_input[0].set_value("second")
            app.text_input[1].set_value("synthetic-password-123")
            app.button[0].click().run()
            self.assertNotIn("Adjudication", app.radio[0].options)
            app.radio[0].set_value("Positive Review").run()
            next(button for button in app.button if button.label == "Mở / tiếp tục phiên review").click().run()
            self.settle(app)
            try:
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual(len(app.get("image")), 1)
                next(button for button in app.button if button.label == "Next Frame →").click().run()
                self.assertEqual(app.number_input[0].value, 1)
                next(button for button in app.button if button.label == "Mark Start").click().run()
                app.number_input[0].set_value(8).run()
                next(button for button in app.button if button.label == "Mark End").click().run()
                next(button for button in app.button if button.label == "Add Event").click().run()
                next(button for button in app.button if button.label == "Save Draft").click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                review = self.repository.own_reviews(self.tokens["second"])[0]
                self.assertEqual(review["status"], "DRAFT")
                self.assertEqual(len(review["payload"]["events"]), 1)
                self.assertEqual(review["payload"]["events"][0]["frame_start"], 1)
                self.assertEqual(review["payload"]["events"][0]["frame_end"], 8)
                stable_id = review["payload"]["events"][0]["event_id"]
                next(button for button in app.button if button.label == "Edit").click().run()
                app.number_input[0].set_value(2).run()
                next(button for button in app.button if button.label == "Mark Start").click().run()
                app.number_input[0].set_value(9).run()
                next(button for button in app.button if button.label == "Mark End").click().run()
                next(button for button in app.button if button.label == "Cập nhật interval").click().run()
                next(button for button in app.button if button.label == "Save Draft").click().run()
                event = self.repository.own_reviews(self.tokens["second"])[0]["payload"]["events"][0]
                self.assertEqual(event["event_id"], stable_id)
                self.assertEqual((event["frame_start"], event["frame_end"]), (2, 9))
                next(button for button in app.button if button.label == "Delete").click().run()
                next(button for button in app.button if button.label == "Save Draft").click().run()
                self.assertEqual(self.repository.own_reviews(self.tokens["second"])[0]["payload"]["events"], [])
                app.radio[0].set_value("Negative Review").run()
                next(button for button in app.button if button.label == "Mở / tiếp tục phiên review").click().run()
                self.settle(app)
                app.number_input[0].set_value(2).run()
                next(button for button in app.button if button.label == "Mark Start").click().run()
                app.number_input[0].set_value(8).run()
                next(button for button in app.button if button.label == "Mark End").click().run()
                next(button for button in app.button if button.label == "Add Event").click().run()
                next(box for box in app.selectbox if box.label == "Quyết định semantic").set_value("CONTAINS_FIRE").run()
                next(item for item in app.text_input if item.label == "Evidence reference").set_value("SYNTHETIC_UI_FIXTURE_ONLY")
                app.checkbox[0].check()
                next(button for button in app.button if button.label == "Submit Review").click().run()
                self.assertFalse(app.error)
                self.submit("first", "NEG_01", [self.event("NEG_01")], "CONTAINS_FIRE")
                next(button for button in app.button if button.label == "Đăng xuất").click().run()
                app.text_input[0].set_value("judge")
                app.text_input[1].set_value("synthetic-password-123")
                app.button[0].click().run()
                app.radio[0].set_value("Adjudication").run()
                self.assertFalse(app.error)
                next(box for box in app.selectbox if box.label == "Quyết định cuối").set_value("ACCEPT").run()
                next(item for item in app.text_area if item.label == "Lý do phân xử").set_value("SYNTHETIC_UI_FIXTURE_ONLY")
                next(item for item in app.text_input if item.label == "Bằng chứng phân xử").set_value("SYNTHETIC_UI_FIXTURE_ONLY")
                next(button for button in app.button if button.label == "Ghi quyết định phân xử").click().run()
                self.assertFalse(app.error)
                self.assertFalse(app.exception)
                next(radio for radio in app.radio if radio.label == "Điều hướng").set_value("Export Report").run()
                next(button for button in app.button if button.label == "Xuất bundle mới trong private directory").click().run()
                self.assertFalse(app.error)
                self.assertFalse(app.exception)
                self.assertTrue(list((self.private / "exports").glob("*/ground_truth_provenance.json")))
            finally:
                next(button for button in app.button if button.label == "Đăng xuất").click().run()
