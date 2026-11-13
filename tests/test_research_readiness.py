"""Research planning never consumes validation/test or invents missing training data."""
import unittest
from src.research_readiness import replay_plan,split_leaks


class TestResearchReadiness(unittest.TestCase):
    def test_replay_train_only_deterministic(self):
        plan=replay_plan(['v1','v2','v3','v4'],['d1','d2'],{'val'})
        self.assertEqual(plan['fraction'],.2)
        self.assertEqual(plan,replay_plan(['v1','v2','v3','v4'],['d1','d2'],{'val'}))

    def test_missing_data_blocks_and_leakage_raises(self):
        self.assertEqual(replay_plan(['v1'],[],set())['status'],'BLOCKED_TRAIN_DATA')
        with self.assertRaises(ValueError): replay_plan(['v1'],['test'],{'test'})

    def test_video_group_leakage(self):
        self.assertEqual(split_leaks([{'video_id':'v','split':'train'},{'video_id':'v','split':'test'}]),['v'])
