"""Unit tests for cold-start popularity, genre round-robin, and content-based onboarding."""


def test_new_user_gets_k_items(cold_start):
    res = cold_start.popular(10)
    assert len(res) == 10
    assert len(set(res)) == 10


def test_genre_round_robin(cold_start):
    res = cold_start.popular(6, genres=["Action", "Comedy"])
    assert len(res) == 6
    assert len(set(res)) == 6


def test_content_based_excludes_liked(cold_start):
    liked = [1, 2, 3]
    res = cold_start.from_liked_items(liked, k=10)
    assert len(res) == 10
    assert all(i not in liked for i in res)


def test_lukewarm_blend_excludes_history(cold_start):
    history = [4, 5]
    warm_ids = [6, 7, 8, 9, 10, 11, 12]
    res = cold_start.blend_lukewarm(warm_ids, history, k=8)
    assert len(res) == 8
    assert all(i not in history for i in res)
