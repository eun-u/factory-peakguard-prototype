import pytest

from phase_f.models.foundation import require_finetune_mode


def test_explicit_zero_shot_flag_is_accepted():
    require_finetune_mode(False)
    require_finetune_mode(None)


@pytest.mark.parametrize('mode', [True, 0, '', 'false', 'unsupported'])
def test_unknown_training_modes_are_rejected(mode):
    with pytest.raises(ValueError, match='Unknown finetuning mode'):
        require_finetune_mode(mode)
