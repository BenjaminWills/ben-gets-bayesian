from exercise import uniform_prior, make_grid, triangular_prior


def test_uniform_prior():
    grid = make_grid(6)
    assert list(uniform_prior(grid)) == [1/6]*6


def test_triangular_prior_odd_length():
    grid = make_grid(5)
    peak = 0.5
    tp = triangular_prior(grid, peak)
    assert list(tp) == [0, 0.25, 0.5, 0.25, 0]
    assert abs(sum(tp) - 1) < 0.01
