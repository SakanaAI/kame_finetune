from collections.abc import Sequence

from kame.models import LMModel


def backfill_oracle_embedding_from_text(moshi_lm: LMModel) -> None:
    moshi_lm.oracle_emb.load_state_dict(moshi_lm.text_emb.state_dict())


def expected_missing_oracle_embedding_keys(moshi_lm: LMModel) -> set[str]:
    return {f"oracle_emb.{key}" for key in moshi_lm.oracle_emb.state_dict().keys()}


def validate_oracle_embedding_checkpoint_load(
    moshi_lm: LMModel,
    *,
    missing_keys: Sequence[str],
    unexpected_keys: Sequence[str],
    context: str,
) -> bool:
    oracle_missing_keys = {key for key in missing_keys if key.startswith("oracle_emb.")}
    non_oracle_missing_keys = sorted(
        key for key in missing_keys if not key.startswith("oracle_emb.")
    )
    unexpected_keys = sorted(unexpected_keys)

    if unexpected_keys or non_oracle_missing_keys:
        details = []
        if unexpected_keys:
            details.append(f"Unexpected keys: {unexpected_keys}.")
        if non_oracle_missing_keys:
            details.append(f"Missing non-oracle_emb keys: {non_oracle_missing_keys}.")
        raise RuntimeError(f"Incompatible state_dict when {context}. {' '.join(details)}")

    expected_oracle_missing_keys = expected_missing_oracle_embedding_keys(moshi_lm)
    if not oracle_missing_keys:
        return False
    if oracle_missing_keys == expected_oracle_missing_keys:
        return True

    raise RuntimeError(
        f"Partially missing oracle_emb weights when {context}. "
        "Expected either no missing oracle_emb keys or the full oracle_emb key set "
        f"{sorted(expected_oracle_missing_keys)}. "
        f"Missing oracle_emb keys: {sorted(oracle_missing_keys)}."
    )
