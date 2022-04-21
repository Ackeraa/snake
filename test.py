def f(score, steps):
    return  0.6 * (score / 100) + 0.4 * (100 * score / ((100 * score - 1) * steps))

def test(a, b, c, d):
    print(f(a, b), f(c, d))

test(1, 3, 1, 1)
