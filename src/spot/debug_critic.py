from spot.decode import CriticAssesInfo
from spot.model import DatasetPredResult
from spot.type_check import PythonType, normalize_type
from spot.utils import *


def check_delta(
    bs_result: DatasetPredResult,
    oracle_result: DatasetPredResult,
    critic_result: DatasetPredResult[CriticAssesInfo],
):
    """
    Check how the critic performance changes inside and outside of the "delta"
    between beam search and oracle reranking result.
    """
    assert_eq(
        len(bs_result.chunks), len(oracle_result.chunks), len(critic_result.chunks)
    )

    def critic_asses_error(ids: list[tuple[int, int]]):
        scores = list[float]()
        targets = list[float]()
        for (cid, lid) in ids:
            einfo = critic_result.extra_info[cid]
            sample_id = int(np.argmax(einfo.candidate_scores))
            asses = einfo.candidate_label_scores[sample_id][lid]
            scores.append(asses)
            info = critic_result.chunks.chunks_info[cid]
            target = float(
                normalize_type(critic_result.predictions[cid][lid])
                == normalize_type(info.types[lid])
            )
            targets.append(target)
        return np.array(scores, dtype=np.float64), np.array(targets, dtype=np.float64)

    # compute the diff between the predictions of the beam search and of the oracle
    diffs = list[tuple[int, int]]()
    n_total_preds = 0
    for i in range(len(bs_result.chunks)):
        for j in range(len(bs_result.predictions[i])):
            n_total_preds += 1
            if bs_result.predictions[i][j] != oracle_result.predictions[i][j]:
                diffs.append((i, j))

    # compute some summary statistics
    diff_ratio = len(diffs) / n_total_preds
    diff_scores, diff_targets = critic_asses_error(diffs)
    all_scores, all_targets = critic_asses_error(
        [
            (i, j)
            for i in range(len(bs_result.chunks))
            for j in range(len(bs_result.predictions[i]))
        ]
    )

    return {
        "diff_ratio": diff_ratio,
        "diff_critic_error": float(np.mean(diff_scores - diff_targets)),
        "all_critic_error": float(np.mean(all_scores - all_targets)),
        "diff_critic_abs_error": float(np.mean(np.abs(diff_scores - diff_targets))),
        "all_critic_abs_error": float(np.mean(np.abs(all_scores - all_targets))),
        "diff_scores_distr": Counter([to_score_range(s) for s in sorted(diff_scores)]),
        "all_scores_distr": Counter([to_score_range(s) for s in sorted(all_scores)]),
    }


def to_score_range(s) -> str:
    s_floor = int(s / 0.1) * 0.1
    return f"{s_floor:.1f}"


def inspect_critic_on_beams(
    result: DatasetPredResult[CriticAssesInfo],
    candidates: list[list[list[PythonType]]],
):
    # track accuracy against the rank in the beam search
    model_by_rank = GroupedAccCounter[int]()
    critic_by_rank = GroupedAccCounter[int]()
    # track critic accuracy against the score predicted by the critic
    # the scores are divided into 10% ranges.
    model_by_score = GroupedAccCounter[str]()
    critic_by_score = GroupedAccCounter[str]()

    for i in range(len(result.chunks)):
        labels = result.chunks.chunks_info[i].types
        einfo = result.extra_info[i]
        for j, cand_preds in enumerate(candidates[i]):
            for p, l, s in zip(cand_preds, labels, einfo.candidate_label_scores[j]):
                model_correct = normalize_type(p) == normalize_type(l)
                critic_correct = (s > 0.5) == model_correct
                model_by_rank.count(j, model_correct, 1)
                critic_by_rank.count(j, critic_correct, 1)
                model_by_score.count(to_score_range(s), model_correct, 1)
                critic_by_score.count(to_score_range(s), critic_correct, 1)

    return {
        "Model acc by rank": model_by_rank.grouped_accs(),
        "Model acc by score": model_by_score.grouped_accs(),
        "Critic acc by rank": critic_by_rank.grouped_accs(),
        "Critic acc by score": critic_by_score.grouped_accs(),
    }
