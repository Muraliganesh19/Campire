#!/usr/bin/env python3
"""Turn the hand-annotated simulation data into videos for the Campire host.

    python simulation/videomaker.py process <delivery folder> <session>
        What ANALYZE runs for a folder from ip_vid. Draws each clip's
        annotations onto its frames, stores the videos in op_vid/<type>, and
        copies them to the host's Output folder for the dashboard.

    python simulation/videomaker.py render [type1 type2 ...]
        Render deliveries into op_vid without the app. With no names it renders
        every ip_vid/type* folder.

    python simulation/videomaker.py
        What ANALYZE runs for a folder of videos. Copies the session chosen with
        UPLOAD FILES from Input to Output.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None  # The host's system Python has neither. See require_opencv().

SIM = Path(__file__).resolve().parent
CLIPS = ('c1', 'c2', 'c3', 'c4')
FPS = 30
MAX_SIDE = 1280  # The Linux host draws 1080x1920 clips smaller in their panels.
NIL = {'bowler_stance': 'NIL', 'delivery': 'NIL', 'wicket': 'NIL'}

# Colours are BGR. Names follow the labels shown on the host's result screen.
STYLE = {
    'ball': ('BALL', (40, 40, 230)),
    'batter': ('BATTER', (225, 120, 30)),
    'bowler': ('BOWLER', (30, 220, 240)),
    'wicket': ('STUMPS', (60, 190, 60)),
}
GUIDE_COLOUR = (230, 230, 230)  # wide-* and bowler-* are guide lines, not objects.


def require_opencv():
    """Rerun under the project's .venv if this Python has no OpenCV."""
    if cv2 is not None:
        return
    name = 'Scripts/python.exe' if os.name == 'nt' else 'bin/python'
    venv_python = SIM.parent / '.venv' / name
    if not venv_python.exists() or Path(sys.prefix).resolve() == venv_python.parents[1].resolve():
        sys.exit(f'Processing needs OpenCV, but {venv_python} cannot provide it. '
                 'Create .venv as described in requirements.txt.')
    sys.exit(subprocess.run([str(venv_python), *sys.argv]).returncode)


def base_dir():
    env = os.environ.get('CAMPIRE_BASE_DIR')
    if env:
        return Path(env)
    return Path(r'E:\campire') if os.name == 'nt' else Path.home() / 'campire'


def read_decisions(path):
    """Return the decisions in path, or None if it is missing, empty or invalid."""
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data else None


def load_shapes(xml_path):
    """Map each frame number to its visible (label, tag, attributes) shapes."""
    shapes = {}
    for track in ET.parse(xml_path).getroot().iter('track'):
        for shape in track:
            if shape.get('outside') != '1':
                entry = (track.get('label'), shape.tag, shape.attrib)
                shapes.setdefault(int(shape.get('frame')), []).append(entry)
    return shapes


def draw_shape(frame, label, tag, attrs, line):
    name, colour = STYLE.get(label, (None, GUIDE_COLOUR))
    angle = float(attrs.get('rotation', 0))

    if tag == 'ellipse':
        cx, cy = float(attrs['cx']), float(attrs['cy'])
        rx, ry = float(attrs['rx']), float(attrs['ry'])
        centre, axes = (round(cx), round(cy)), (round(rx), round(ry))
        cv2.ellipse(frame, centre, axes, angle, 0, 360, colour, line, cv2.LINE_AA)
        left, top = cx - rx, cy - ry
    else:
        x1, y1 = float(attrs['xtl']), float(attrs['ytl'])
        x2, y2 = float(attrs['xbr']), float(attrs['ybr'])
        rect = (((x1 + x2) / 2, (y1 + y2) / 2), (x2 - x1, y2 - y1), angle)
        corners = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(frame, [corners], True, colour, line, cv2.LINE_AA)
        left, top = corners[:, 0].min(), corners[:, 1].min()

    if name:
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, line * 0.3
        (width, height), _ = cv2.getTextSize(name, font, scale, line)
        pad = line * 2
        x = max(int(left), 0)
        y = max(int(top), height + pad * 2)
        cv2.rectangle(frame, (x, y - height - pad * 2), (x + width + pad * 2, y), colour, -1)
        cv2.putText(frame, name, (x + pad, y - pad), font, scale, (255, 255, 255), line, cv2.LINE_AA)


def clip_frames(clip_dir):
    images = clip_dir / 'images'
    return sorted(p for p in images.iterdir() if p.suffix.lower() == '.png') if images.is_dir() else []


def render_clip(clip_dir, out_path):
    frames = clip_frames(clip_dir)
    shapes = load_shapes(clip_dir / 'annotations.xml')

    height, width = cv2.imread(str(frames[0])).shape[:2]
    line = max(2, round(height / 540))
    scale = min(1, MAX_SIDE / max(width, height))
    size = (round(width * scale), round(height * scale))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, FPS, size)
    if not writer.isOpened():
        raise OSError(f'Could not write {out_path}')

    for path in frames:
        image = cv2.imread(str(path))
        number = int(re.search(r'(\d+)$', path.stem).group(1))
        for label, tag, attrs in shapes.get(number, []):
            draw_shape(image, label, tag, attrs, line)
        if scale < 1:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        writer.write(image)
    writer.release()
    return f'  {out_path.name}: {len(frames)} frames at {size[0]}x{size[1]}'


def render_delivery(source):
    """Render one ip_vid delivery into op_vid and return the folder it wrote."""
    for clip in CLIPS:
        if not clip_frames(source / clip) or not (source / clip / 'annotations.xml').is_file():
            sys.exit(f'{source / clip} needs an images folder and annotations.xml')

    print(f'{source.name}:', flush=True)
    target = SIM / 'op_vid' / source.name
    target.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=len(CLIPS)) as pool:
        jobs = [pool.submit(render_clip, source / c, target / f'{c}.mp4') for c in CLIPS]
        for job in jobs:
            print(job.result(), flush=True)

    decisions = read_decisions(source / 'decisions.json')
    if decisions is None:
        print('  decisions.json is empty, so the host will show NIL', flush=True)
        decisions = {'session_id': source.name, **NIL}
    (target / 'decisions.json').write_text(json.dumps(decisions, indent=2) + '\n', encoding='utf-8')
    return target


def process(folder, session):
    require_opencv()
    target = render_delivery(Path(folder))
    output = base_dir() / 'Output'
    output.mkdir(parents=True, exist_ok=True)
    for number, clip in enumerate(CLIPS, start=1):
        shutil.copy2(target / f'{clip}.mp4', output / f'{session}_cam{number}.mp4')
    shutil.copy2(target / 'decisions.json', output / f'{session}.json')


def render(names):
    require_opencv()
    root = SIM / 'ip_vid'
    names = names or sorted(p.name for p in root.glob('type*') if p.is_dir())
    if not names:
        sys.exit(f'No type* folders in {root}')
    for name in names:
        render_delivery(root / name)


def hand_off():
    source, target = base_dir() / 'Input', base_dir() / 'Output'
    videos = sorted(source.glob('fileip*_cam*.mp4'))
    if not videos:
        sys.exit('That folder has no videos and no c1 to c4 clips. '
                 'Choose a delivery such as simulation/ip_vid/type1 with UPLOAD FILES.')

    target.mkdir(parents=True, exist_ok=True)
    for video in videos:
        shutil.copy2(video, target / video.name)

    session = videos[0].name.split('_cam')[0]
    if read_decisions(source / f'{session}.json') is None:
        decisions = {'session_id': session, **NIL}
        (target / f'{session}.json').write_text(json.dumps(decisions), encoding='utf-8')
    print(f'{session}: {len(videos)} videos ready')


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        hand_off()
    elif args[0] == 'process' and len(args) == 3:
        process(args[1], args[2])
    elif args[0] == 'render':
        render(args[1:])
    else:
        sys.exit(__doc__)
