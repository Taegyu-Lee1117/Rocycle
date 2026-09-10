import os

import sounddevice as sd


def resolve_input_device():
    override = os.environ.get("VOICE_MIC_DEVICE")
    if override:
        return int(override) if override.isdigit() else override
    try:
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and int(round(dev["default_samplerate"])) == 16000:
                return idx
    except Exception:
        pass
    return None
