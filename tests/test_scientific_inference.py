"""Evidence schema, VFR grid, coverage failures and independent-test guard."""
import unittest
from src.scientific_inference import sample_indices,coverage,require_independent_test


class TestScientificInference(unittest.TestCase):
    def test_cfr_and_vfr_source_sampling(self):
        self.assertEqual(sample_indices(list(range(30)),'1/30'),[0,6,12,18,24])
        self.assertEqual(sample_indices([100,101,103,106,107],'1/10'),[0,2,3])

    def test_sparse_source_never_duplicated(self):
        self.assertEqual(sample_indices([0,1,2],'1/2'),[0,1,2])
        with self.assertRaises(ValueError):sample_indices([0,0],'1/10')

    def rows(self):
        return [{'run_id':'synthetic-test-only','processing_status':'OK','frame_index':i,'source_pts_ticks':i,'time_base':'1/5','detections':[]} for i in range(3)]

    def test_empty_detections_are_real_coverage_records(self):
        self.assertTrue(coverage(self.rows(),[0,1,2],[0,1,2],'1/5',{'run_id':'synthetic-test-only'})['complete'])

    def test_partial_and_duplicate_evidence_rejected(self):
        for rows in (self.rows()[:2],self.rows()+self.rows()[:1]):
            with self.assertRaises(ValueError):coverage(rows,[0,1,2],[0,1,2],'1/5',{})

    def test_checkpoint_and_source_binding_rejected(self):
        for binding in ({'checkpoint_sha256':'required'},{'source_sha256':'required'},{'run_id':'another-run'}):
            with self.assertRaises(ValueError):coverage(self.rows(),[0,1,2],[0,1,2],'1/5',binding)

    def test_pts_tampering_rejected(self):
        rows=self.rows();rows[1]['source_pts_ticks']=99
        with self.assertRaises(ValueError):coverage(rows,[0,1,2],[0,1,2],'1/5',{})

    def test_contaminated_test_and_missing_lock_rejected(self):
        with self.assertRaisesRegex(ValueError,'CONTAMINATION'):require_independent_test({'independent_test_valid':False},{'status':'LOCKED'})
        with self.assertRaisesRegex(ValueError,'NOT_VALIDATION_LOCKED'):require_independent_test({'independent_test_valid':True},{'status':'PROVISIONAL'})
        require_independent_test({'independent_test_valid':True,'validation_scope_hash':'a'},{'status':'LOCKED','validation_scope_hash':'a'})

    def test_zero_false_alarm_has_nonzero_ci_upper(self):
        from scripts.evaluate_week4_final_scientific import measure
        gt=[dict(video_id='synthetic-P',camera_id='BENCH_P',label='FIRE',start_sec=2.,end_sec=3.)]
        entries=[dict(video_id='synthetic-P',kind='positive',duration_seconds=10),dict(video_id='synthetic-N',kind='negative',duration_seconds=3600)]
        result=measure(entries,gt,gt)
        self.assertEqual(result['metrics']['false_alarms_per_hour'],0)
        self.assertGreater(result['metrics']['poisson_ci_95'][1],0)
        self.assertFalse(result['independent_test_valid'])

    def test_missing_exposure_not_zero_far(self):
        from scripts.evaluate_week4_final_scientific import measure
        gt=[dict(video_id='synthetic-P',camera_id='BENCH_P',label='FIRE',start_sec=2.,end_sec=3.)]
        result=measure([dict(video_id='synthetic-P',kind='positive',duration_seconds=10)],gt,gt)
        self.assertIsNone(result['metrics']['false_alarms_per_hour'])
        self.assertIsNone(result['metrics']['poisson_ci_95'])
