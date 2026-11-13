"""Synthetic-only audited PTS and provisional split UI integration tests."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import PropertyMock, patch

from gt_studio_fixtures import StudioCase
from ground_truth_studio.config import Settings
from ground_truth_studio.integrity_service import sha256
from ground_truth_studio.review_repository import ReviewRepository
from ground_truth_studio.video_service import VideoService
from src.preannotation import PROVISIONAL, canonical_hash, propose_split
from src.video_audit_v2 import audit_video


class TestPreannotationStudio(StudioCase):
    def prepare_candidate(self, offset=0.):
        if not self.settings.ffmpeg or not self.settings.ffprobe:
            self.skipTest('FFmpeg/FFprobe required for actual synthetic decode integration')
        video=self.root/'datasets/negative_videos/VN_POS_01.mp4'
        subprocess.run([self.settings.ffmpeg,'-nostdin','-v','error','-i',str(self.root/'FireSmokeLabeler/videos/source.avi'),'-c:v','libx264','-pix_fmt','yuv420p','-output_ts_offset',str(offset),str(video)],capture_output=True,check=True,timeout=30)
        digest=sha256(video)
        entry={'video_id':'VN_POS_01','relative_path':'datasets/negative_videos/VN_POS_01.mp4','sha256':digest}
        folder=self.root/'runs/benchmarks/vn_pos_preannotation/fixture/records/VN_POS_01'
        record=audit_video(entry,self.root,folder,self.settings.ffmpeg,self.settings.ffprobe,30)
        self.assertTrue(record['technical_eligible'])
        catalog=self.root/'data/video_catalog_local_v2.json';catalog.write_text(json.dumps({'videos':[entry]}))
        rows=propose_split([{'video_id':'VN_POS_01','source_sha256':digest,'scene_group_id':'synthetic-scene','legacy_splits':['test'],'duration_seconds':2.,'decode_status':'NORMAL_EOF'}])
        rows[0].update(audit_record_path=(folder/'record.json').relative_to(self.root).as_posix(),audit_record_sha256=sha256(folder/'record.json'),resolution=[64,48],codec='h264',pix_fmt='yuv420p',model_predictions=['MUST_NOT_LEAK_TO_ANNOTATOR'])
        value={'status':PROVISIONAL,'input_catalog_sha256':sha256(catalog),'videos':rows}
        value['manifest_sha256']=canonical_hash(value)
        manifest=self.root/'data/vn_pos_preannotation_split_manifest.json';manifest.write_text(json.dumps(value))
        (self.root/'data/vn_pos_split_provenance.json').write_text(json.dumps({'manifest_json_sha256':sha256(manifest)}))
        self.repository=ReviewRepository(self.settings)
        return self.repository.video('VN_POS_01'),folder

    def test_audited_pts_and_explicit_duration_work_without_fresh_ffprobe(self):
        frozen=self.root/'data/ground_truth/events_v1.1_snapshot.jsonl';before=sha256(frozen)
        video,_=self.prepare_candidate()
        with patch.object(Settings,'ffprobe',new_callable=PropertyMock,return_value=None):
            metadata=self.videos.inspect(video)
        self.assertEqual(metadata['frame_count'],20)
        self.assertEqual(metadata['timestamp_provenance'],'SOURCE_PTS')
        self.assertIn('EXPLICIT_FRAME_DURATION',metadata['duration_provenance'])
        self.assertAlmostEqual(metadata['duration'],2.)
        for index in [0,15,3]:
            image,point=self.videos.frame(video,index)
            self.assertEqual(point['frame_index'],index)
            self.assertAlmostEqual(float(image.mean()),index*10,delta=5)
        self.assertEqual(sha256(frozen),before)

    def test_changed_audit_timeline_is_blocked(self):
        video,folder=self.prepare_candidate()
        with (folder/'timeline.stdout').open('ab') as stream:stream.write(b'\n')
        service=VideoService(self.settings);self.addCleanup(service.close)
        with self.assertRaisesRegex(ValueError,'timeline changed'):service.inspect(video)

    def test_nonzero_raw_mp4_clock_maps_to_correct_source_frame(self):
        video,_=self.prepare_candidate(offset=.1)
        metadata=self.videos.inspect(video)
        self.assertAlmostEqual(metadata['source_pts'][0],.1)
        self.assertEqual(self.videos.index_at_time(metadata,self.videos.source_time_from_playback(metadata,.1)),0)
        self.assertAlmostEqual(self.videos.playback_time_from_source(metadata,1.5),1.6)
        image,point=self.videos.frame(video,self.videos.index_at_time(metadata,self.videos.source_time_from_playback(metadata,1.6)))
        self.assertEqual(point['frame_index'],15);self.assertAlmostEqual(float(image.mean()),150,delta=5)
        with self.assertRaises(ValueError):self.videos.source_time_from_playback(metadata,float('nan'))

    def test_candidate_ui_shows_only_provisional_split_and_blank_draft(self):
        from streamlit.testing.v1 import AppTest
        self.prepare_candidate()
        config=self.private/'candidate_ui.json';config.write_text(json.dumps({'repo_root':str(self.root),'private_dir':str(self.private)}))
        entry=Path(__file__).resolve().parent.parent/'ground_truth_studio/app.py'
        with patch.dict('os.environ',{'GT_STUDIO_CONFIG':str(config),'GT_STUDIO_PRIVATE_DIR':str(self.private)}):
            app=AppTest.from_file(entry,default_timeout=60).run()
            app.text_input[0].set_value('second');app.text_input[1].set_value('synthetic-password-123')
            app.button[0].click().run();app.radio[0].set_value('VN_POS Candidate Review').run()
            self.assertFalse(app.exception);self.assertFalse(app.error)
            captions=' '.join(item.value for item in app.caption)
            self.assertIn('TEST_INDEPENDENCE_NOT_VERIFIED',captions)
            self.assertNotIn('MUST_NOT_LEAK_TO_ANNOTATOR',captions)
            next(b for b in app.button if b.label=='Mở / tiếp tục phiên review').click().run()
            deadline=time.monotonic()+15
            while any('PREPARING_SOURCE' in item.value for item in app.info):
                self.assertLess(time.monotonic(),deadline);time.sleep(.05);app.run()
            self.assertFalse(app.exception);self.assertFalse(app.error)
            draft=self.repository.own_reviews(self.tokens['second'])[0]
            self.assertEqual(draft['status'],'DRAFT');self.assertEqual(draft['payload']['events'],[])
            self.assertNotIn('model_predictions',draft['payload'])
            next(b for b in app.button if b.label=='Đăng xuất').click().run()
