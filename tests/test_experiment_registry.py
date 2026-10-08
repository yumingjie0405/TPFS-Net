from tools._io import best_epoch_from_log, resolve_checkpoint


def test_best_checkpoint_follows_log_not_latest(tmp_path):
    (tmp_path / 'runs-138-86388.pt').touch()
    (tmp_path / 'runs-140-87640.pt').touch()
    (tmp_path / 'log.txt').write_text('Epoch number: 138\n')

    assert best_epoch_from_log(tmp_path) == 138
    assert resolve_checkpoint(tmp_path).endswith('runs-138-86388.pt')
    assert resolve_checkpoint(tmp_path, policy='latest').endswith('runs-140-87640.pt')
