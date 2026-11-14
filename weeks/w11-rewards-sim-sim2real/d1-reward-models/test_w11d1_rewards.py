import torch

import reward_data
from w11d1_rewards import (pairwise_loss, progress_pairs, spearman, threshold_for_precision, train_progress_model,
                           train_success_classifier)


def test_classifier_learns():
    tr, va = reward_data.classifier_data(seed=0), reward_data.classifier_data(seed=1)
    clf = train_success_classifier(tr["x"], tr["y"], seed=0)
    with torch.no_grad():
        out = clf(va["x"])
    assert out.shape == (len(va["x"]), 1)
    acc = ((out.squeeze(-1) > 0) == va["y"].bool()).float().mean().item()
    assert acc > 0.95, f"accuracy {acc:.3f}"


def test_threshold_for_precision():
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    labels = torch.tensor([1, 1, 0, 1, 0, 0])
    assert abs(threshold_for_precision(scores, labels, 1.0) - 0.8) < 1e-6
    assert abs(threshold_for_precision(scores, labels, 0.75) - 0.6) < 1e-6
    assert abs(threshold_for_precision(scores, labels, 0.5) - 0.4) < 1e-6
    assert threshold_for_precision(torch.tensor([0.9, 0.1]), torch.tensor([0, 1]), 0.9) == float("inf")
    # ties must be taken together: both 0.5 scores, one positive -> precision 0.5 at t = 0.5
    assert threshold_for_precision(torch.tensor([0.5, 0.5]), torch.tensor([1, 0]), 0.9) == float("inf")


def test_pairs_stay_inside_one_episode():
    ep = reward_data.expert_episodes(n=30, seed=0)
    ia, ib, y = progress_pairs(ep["episode"], ep["t"], 2000, torch.Generator().manual_seed(0))
    assert torch.equal(ep["episode"][ia], ep["episode"][ib])
    assert (ep["t"][ia] != ep["t"][ib]).all()
    assert torch.equal(y, (ep["t"][ib] > ep["t"][ia]).float())
    assert 0.4 < y.mean() < 0.6
    assert ep["episode"][ia].unique().numel() > 20


def test_pairwise_loss_and_spearman():
    l_good = pairwise_loss(torch.tensor([0.0]), torch.tensor([5.0]), torch.tensor([1.0]))
    l_bad = pairwise_loss(torch.tensor([0.0]), torch.tensor([5.0]), torch.tensor([0.0]))
    assert l_good < 0.01 and l_bad > 4.9
    assert torch.isclose(pairwise_loss(torch.zeros(3), torch.zeros(3), torch.ones(3)), torch.log(torch.tensor(2.0)))
    assert abs(spearman(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([1.0, 8.0, 27.0, 1000.0])) - 1) < 1e-6
    assert abs(spearman(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0])) + 1) < 1e-6


def test_progress_model_tracks_distance():
    ep, te = reward_data.expert_episodes(seed=0), reward_data.expert_episodes(seed=1)
    model = train_progress_model(ep["x"], ep["episode"], ep["t"], seed=0)
    with torch.no_grad():
        rho = spearman(model(te["x"]).squeeze(-1), -te["dist"])
    assert rho > 0.85, f"Spearman {rho:.2f}"
