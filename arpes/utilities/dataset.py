# These utilities can be made smarter

def map_scans(metadata: dict):
    return [scan for scan in metadata if 'type' in scan['note'] and scan['note']['type'] == 'map']

def hv_scans(metadata: dict):
    return [scan for scan in metadata if 'type' in scan['note'] and scan['note']['type'] == 'hv']