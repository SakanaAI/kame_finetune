import finetune


def test_compute_num_update_steps_per_epoch_uses_local_batches():
    assert finetune._compute_num_update_steps_per_epoch(50, 2) == 25
    assert finetune._compute_num_update_steps_per_epoch(51, 2) == 26


def test_compute_resume_batch_offset_uses_local_batch_count():
    assert (
        finetune._compute_resume_batch_offset(
            13,
            gradient_accumulation_steps=2,
            num_local_batches_per_epoch=50,
        )
        == 26
    )


def test_compute_resume_batch_offset_wraps_at_epoch_boundary():
    assert (
        finetune._compute_resume_batch_offset(
            26,
            gradient_accumulation_steps=2,
            num_local_batches_per_epoch=50,
        )
        == 2
    )
