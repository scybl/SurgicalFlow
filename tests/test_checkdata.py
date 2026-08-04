from checkdata import (
    analyze_each_video,
    compress_sequence,
    load_phase_file,
    plot_pattern_distribution,
)


def test_load_phase_file_skips_header_and_unknown_phase(tmp_path):
    phase_file = tmp_path / "video01-phase.txt"
    phase_file.write_text(
        "\n".join(
            [
                "Frame Phase",
                "0 Preparation",
                "1 Preparation",
                "2 UnknownPhase",
                "3 CalotTriangleDissection",
            ]
        ),
        encoding="utf-8",
    )

    assert load_phase_file(phase_file) == [0, 0, 1]


def test_compress_sequence_removes_consecutive_duplicates():
    assert compress_sequence([0, 0, 1, 1, 2, 0, 0]) == (0, 1, 2, 0)
    assert compress_sequence([]) == tuple()


def test_analyze_each_video_groups_identical_patterns(tmp_path):
    first = tmp_path / "video01-phase.txt"
    first.write_text(
        "Frame Phase\n0 Preparation\n1 CalotTriangleDissection\n",
        encoding="utf-8",
    )
    second = tmp_path / "video02-phase.txt"
    second.write_text(
        "Frame Phase\n0 Preparation\n1 CalotTriangleDissection\n",
        encoding="utf-8",
    )

    video2pattern, pattern2videos = analyze_each_video(tmp_path)

    assert video2pattern == {
        "video01": (0, 1),
        "video02": (0, 1),
    }
    assert pattern2videos[(0, 1)] == ["video01", "video02"]


def test_empty_pattern_distribution_skips_plot(capsys):
    assert plot_pattern_distribution({}) is False
    assert "No valid phase patterns found" in capsys.readouterr().out
