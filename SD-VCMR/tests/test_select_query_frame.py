from fractions import Fraction

from scripts.select_query_frame import nearest_grid_indices


def test_nearest_in_window_vfr_and_tie():
    # The middle grid tick is equidistant from two in-window presentation times.
    times = [Fraction(n, 6) for n in (0, 1, 3, 4)]
    assert nearest_grid_indices(times, Fraction(0), Fraction(1)) == [0, 1, 3]


def test_low_fps_deduplicates_repeated_nearest_frames():
    times = [Fraction(1), Fraction(19, 10)]
    assert nearest_grid_indices(times, Fraction(1), Fraction(2)) == [0, 1]
    assert nearest_grid_indices([Fraction(1)], Fraction(1), Fraction(4, 3)) == [0]



def test_decoder_keeps_only_presentation_times_in_half_open_interval(tmp_path):
    import pytest
    from PIL import Image
    from scripts.select_query_frame import decode_candidates

    av = pytest.importorskip('av')
    path = tmp_path / 'example.mkv'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('ffv1', rate=4)
        stream.width, stream.height = 32, 32
        stream.pix_fmt = 'bgr0'
        for value in range(5):
            frame = av.VideoFrame.from_image(Image.new('RGB', (32, 32), (value, 0, 0)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    selected = list(decode_candidates(path, Fraction(1, 4), Fraction(3, 4)))
    assert [time for time, _ in selected] == [Fraction(1, 4), Fraction(1, 2)]
    assert [frame.getpixel((0, 0))[0] for _, frame in selected] == [1, 2]
