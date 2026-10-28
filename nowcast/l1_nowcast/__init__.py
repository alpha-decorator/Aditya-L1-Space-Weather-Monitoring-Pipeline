"""
l1_nowcast - streaming solar flare nowcasting for Aditya-L1 (SoLEXS + HEL1OS)
and GOES-R XRS.

    from l1_nowcast import NowcastEngine, goes_channels, GOESLiveSource
    eng = NowcastEngine(goes_channels(), cadence_s=60)
    for sample in GOESLiveSource():
        record, messages = eng.update(sample)
"""

from .background import AdaptiveBackground, RollingWindow
from .detector import (
    DECAY, FLARE, QUIET, STATE_NAMES,
    ChannelConfig, ChannelDetector, EventFuser, FlareEvent, FusedFlare,
    NowcastEngine, NowcastResult, aditya_channels, class_letter, class_to_flux,
    fit_counts_to_flux, goes_channels, goes_class, run_batch,
)
from .sources import (
    DirectoryWatchSource, FrameSource, GOESLiveSource, Sample, StreamAligner,
    align_frames, available_days, download_goes_archive, frame_to_samples,
    load_goes_avg1m, load_goes_flare_summary, merge_sources, missing_reference_days, parse_swpc_xrays,
    read_hel1os_csv, read_ogip_lightcurve, read_solexs_csv, regularize, to_epoch_s,
)
from .validate import evaluate, match_events, tune_channel

__version__ = "1.0.0"
