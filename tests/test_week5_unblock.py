"""Incremental audit contracts, using synthetic data only."""
from pathlib import Path
from unittest.mock import patch

from scripts.audit_week5_unblock import baseline_membership


def test_checkpoint_dataset_path_is_not_membership_evidence(tmp_path: Path):
    checkpoint=tmp_path/'synthetic.pt';checkpoint.write_bytes(b'synthetic')
    with patch('scripts.audit_week5_unblock.torch.load',return_value={'train_args':{'data':'data_dfire.yaml'},'epoch':-1}):
        result=baseline_membership(checkpoint)
    assert result['status']=='TRAINING_MEMBERSHIP_NOT_AUTHENTICATED'
    assert result['independent_evaluation_claimed'] is False
    assert result['fallback_protocol']['reuse_historical_test_for_training'] is False
