from routers.video import build_dubbing_filter, build_media_url


def test_build_dubbing_filter_contains_mix_and_volume_controls():
    filter_text = build_dubbing_filter(original_gain=0.18, dub_gain=1.0, original_has_audio=True)
    assert 'volume=0.18' in filter_text
    assert 'volume=1.0' in filter_text
    assert 'amix=' in filter_text


def test_build_dubbing_filter_handles_video_without_original_audio():
    filter_text = build_dubbing_filter(original_gain=0.18, dub_gain=1.0, original_has_audio=False)
    assert 'anullsrc' in filter_text
    assert 'volume=1.0' in filter_text
    assert 'amix=' in filter_text


def test_build_media_url_encodes_special_chars_and_hashes():
    url = build_media_url('English conversation – have a coffee with Tim ☕️ #shorts_shd4avQjheM.mp4')
    assert url.startswith('/media/')
    assert '%23shorts' in url
    assert '%20' in url
    assert '☕' not in url
