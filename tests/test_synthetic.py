from datetime import date

from foundrate.synthetic import Shift, World, WorldConfig, generate_order_lines

CFG = WorldConfig(n_stores=6, n_items=40, lines_per_day=400, seed=1)


def test_deterministic_regardless_of_range():
    w = World(CFG)
    whole = generate_order_lines(w, date(2026, 1, 1), 3)
    day2 = generate_order_lines(w, date(2026, 1, 2), 1)
    assert whole[whole.order_date == "2026-01-02"].reset_index(drop=True).equals(day2)


def test_shift_lowers_found_rate_only_for_target_store():
    w = World(CFG)
    start = date(2026, 1, 1)
    shift = Shift(store_id=2, start=start, logit_delta=3.0)
    base = generate_order_lines(w, start, 10)
    shifted = generate_order_lines(w, start, 10, (shift,))
    for store in range(6):
        b = base[base.store_id == store].found.mean()
        s = shifted[shifted.store_id == store].found.mean()
        if store == 2:
            assert s < b - 0.15
        else:
            assert abs(s - b) < 1e-9
