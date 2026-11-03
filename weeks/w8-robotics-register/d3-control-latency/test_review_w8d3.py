from review_w8d3 import PID


def test_review_derivative_units():
    p = PID(0.0, 0.0, 1.0, 0.01)
    p.step(0.0)
    assert abs(p.step(0.5) - 50.0) < 1e-9, "derivative of a 0.5 change over 10 ms is 50 per second"
