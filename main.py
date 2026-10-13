"""geolip-nano-beatrix-trainer: the entry point for the editor's run button. The work is done by the modules:

    python -m geolip.nano_beatrix.bed_train --out=<out> --arm=smr-fixed --seed=1 --ctx=4096 --mb=8 --steps=12000 --plant=1.0 --records=8
    python -m geolip.nano_beatrix.bed_train --out=<out> --arm=altn-fixed --seed=1 --frames=dyn --plant=1.0 --sizes=32,128,256
    python -m geolip.nano_beatrix.bed_table <out>/bed --pairs
    python -m geolip.nano_beatrix.bed_corpus --out=<out>

This file prints the version, the frame sizes and the registers, and the usage above."""
from geolip.nano_beatrix import __version__, bed_frames

if __name__ == "__main__":
    print(f"geolip-nano-beatrix-trainer {__version__}")
    print("frame sizes:", bed_frames.FRAME_SIZES, "| registers:", list(bed_frames.REGISTERS), "| layouts:", list(bed_frames.LAYOUTS))
    print(__doc__)
